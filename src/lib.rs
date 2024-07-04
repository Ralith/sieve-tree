#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use core::{array, fmt, mem};

use arrayvec::ArrayVec;
use slab::Slab;

mod cell_coords;
use cell_coords::{cell_extent, CellCoords, CellsWithin};

mod tree_bounds;
use tree_bounds::TreeBounds;

mod traversal;
use traversal::Intersections;

/// A `DIM`-dimensional spatial search tree
///
/// Each tree node owns a grid of `2.pow(GRID_EXPONENT).pow(DIM)` cells. Increasing `GRID_EXPONENT`
/// makes the tree less sparse, which accelerates random access by reducing indirection in exchange
/// for exponentially increased memory requirements and less precise balancing.
#[derive(Debug)]
pub struct SieveTree<const DIM: usize, const GRID_EXPONENT: u32, T> {
    /// Length of a level 0 node edge in world space
    granularity: f64,
    root: Option<Root<DIM, GRID_EXPONENT>>,
    elements: Slab<Element<T>>,
}

impl<const DIM: usize, const GRID_EXPONENT: u32, T> SieveTree<DIM, GRID_EXPONENT, T> {
    /// Create an empty tree
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty tree with a custom minimum partition size
    ///
    /// Should be less than the diameter of the bounding sphere of your smallest values, or the
    /// smallest inter-object distance the tree should be able to discriminate, whichever is
    /// greater.
    ///
    /// The tree is able to discriminate positions separated by at most 2^64 multiplied by this
    /// value, so erring on the small side is typically safe.
    ///
    /// The default is 0.01. If world space is measured in meters, this allows partitioning of
    /// objects at centimeter resolution while remaining able to index the entire solar system.
    pub fn with_granularity(granularity: f64) -> Self {
        Self {
            granularity,
            ..Self::default()
        }
    }

    /// Insert `value` into the best existing location in the tree, returning an ID that can be used
    /// to access it directly
    pub fn insert(&mut self, bounds: Bounds<DIM>, value: T) -> usize {
        let id = self.elements.insert(Element {
            value,
            next: None.into(),
        });

        let insert = InsertPoint::find(&mut self.root, self.granularity, bounds);
        link(
            &mut self.elements,
            insert.node,
            insert.cell,
            id,
            insert.sieved,
        );

        id
    }

    /// Insert `value` with `bounds`, splitting nodes if necessary to preserve `elements_per_cell`
    ///
    /// Nodes modified by this method will not need to be [`balance`](Self::balance)d. Prefer this
    /// method over [`insert`](Self::insert) followed by [`balance`](Self::balance) when
    /// intermittently inserting small numbers of elements into trees that will be traversed
    /// heavily. Prefer calling [`balance`](Self::balance) when a large proportion of the tree has
    /// been freshly [`insert`](Self::insert)ed or [`update`](Self::update)d
    pub fn insert_and_balance(
        &mut self,
        bounds: Bounds<DIM>,
        value: T,
        elements_per_cell: usize,
        get_bounds: impl FnMut(&T) -> Bounds<DIM>,
    ) -> usize {
        let id = self.elements.insert(Element {
            value,
            next: None.into(),
        });

        let insert = InsertPoint::find(&mut self.root, self.granularity, bounds);
        link(
            &mut self.elements,
            insert.node,
            insert.cell,
            id,
            insert.sieved,
        );

        // Sieved values are always stored on internal nodes, which store *only* sieved values and
        // hence cannot be subdivided. Hence, this insert can only have introduced new balancing
        // work if it was unsieved.
        if !insert.sieved {
            balance_node(
                insert.node,
                insert.node_level,
                insert.embedding,
                self.granularity,
                &mut self.elements,
                elements_per_cell,
                get_bounds,
            );
        }

        id
    }

    /// Update the bounds of the value associated with `id`
    ///
    /// Similar to `remove` followed by `insert`, but preserves identity and does less work for
    /// small changes.
    pub fn update(&mut self, id: usize, old: Bounds<DIM>, new: Bounds<DIM>) {
        let Some(root) = &mut self.root else {
            panic!("tried to update an element in an empty tree");
        };
        root.ensure_origin_precedes(self.granularity, &new);
        let old = root.embedding.bounds_from_world(self.granularity, &old);
        let old_coords = old.node_location::<GRID_EXPONENT>();
        let new = root.embedding.bounds_from_world(self.granularity, &new);
        let new_coords = new.node_location::<GRID_EXPONENT>();
        let ancestor = old_coords.smallest_common_ancestor(&new_coords);
        let (node, level) = find_smallest_parent(&mut root.node, &mut root.coords, ancestor);

        // Remove from old location
        {
            let (node, level) = find_smallest_existing_parent(level, node, old_coords);
            let cell = grid_index_at_level::<DIM, GRID_EXPONENT>(old.min, level);
            unlink(
                &mut self.elements,
                node,
                cell,
                id,
                level == old_coords.level,
            );
        }

        // Insert into new location. Because `ancestor` was created if necessary, we know that a
        // suitable node already exists.
        let (node, level) = find_smallest_existing_parent(level, node, new_coords);
        let cell = grid_index_at_level::<DIM, GRID_EXPONENT>(new.min, level);
        link(
            &mut self.elements,
            node,
            cell,
            id,
            new_coords.level == level,
        )
    }

    /// Recursively split cells with more than `elements_per_cell` unsieved elements
    ///
    /// Call after large numbers of `insert`s or `update`s to maintain consistent search
    /// performance.
    pub fn balance(&mut self, elements_per_cell: usize, get_bounds: impl FnMut(&T) -> Bounds<DIM>) {
        let Some(ref mut root) = self.root else {
            return;
        };
        balance_node(
            &mut root.node,
            root.coords.level,
            &root.embedding,
            self.granularity,
            &mut self.elements,
            elements_per_cell,
            get_bounds,
        );
    }

    /// Remove the value associated with `id`
    pub fn remove(&mut self, id: usize, bounds: Bounds<DIM>) -> T {
        let root = self.root.as_mut().unwrap();
        let bounds = root.embedding.bounds_from_world(self.granularity, &bounds);
        let target = bounds.node_location::<GRID_EXPONENT>();
        // A value is guaranteed to be stored in the smallest existing node permitted for it, because:
        // - `insert` only introduces nodes that are siblings of or larger than the root
        // - `balance` always moves all possible elements into newly created child nodes
        let (node, level) =
            find_smallest_existing_parent(root.coords.level, &mut root.node, target);
        let cell = grid_index_at_level::<DIM, GRID_EXPONENT>(bounds.min, level);
        unlink(&mut self.elements, node, cell, id, level == target.level);
        let elt = self.elements.remove(id);
        elt.value
    }

    pub fn bounds(&self) -> Option<Bounds<DIM>> {
        self.root
            .as_ref()
            .map(|root| root.world_bounds(self.granularity))
    }

    /// Borrow the value associated with `id`
    pub fn get(&self, id: usize) -> &T {
        &self.elements.get(id).unwrap().value
    }

    /// Uniquely borrow the value associated with `id`
    pub fn get_mut(&mut self, id: usize) -> &mut T {
        &mut self.elements.get_mut(id).unwrap().value
    }

    /// Traverse all elements that might intersect with `bounds`
    pub fn intersections(&self, bounds: Bounds<DIM>) -> Intersections<'_, DIM, GRID_EXPONENT, T> {
        let bounds = self
            .root
            .as_ref()
            .map(|x| x.embedding.bounds_from_world(self.granularity, &bounds))
            .unwrap_or(TreeBounds {
                min: [0; DIM],
                max: [0; DIM],
            });
        Intersections::new(bounds, &self.elements, self.root.as_ref())
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.elements.iter().map(|(i, x)| (i, &x.value))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut T)> {
        self.elements.iter_mut().map(|(i, x)| (i, &mut x.value))
    }
}

impl<const DIM: usize, const GRID_EXPONENT: u32, T> Default for SieveTree<DIM, GRID_EXPONENT, T> {
    fn default() -> Self {
        Self {
            // 1cm if world space is in meters. Big enough for the solar system.
            granularity: 0.01,
            root: None,
            elements: Slab::default(),
        }
    }
}

struct InsertPoint<'a, const DIM: usize, const GRID_EXPONENT: u32> {
    embedding: &'a Embedding<DIM>,
    node: &'a mut Node<DIM, GRID_EXPONENT>,
    node_level: u32,
    cell: usize,
    /// Whether this is the smallest cell (the lowest level) a value with these bounds can be stored
    /// in
    sieved: bool,
}

impl<'a, const DIM: usize, const GRID_EXPONENT: u32> InsertPoint<'a, DIM, GRID_EXPONENT> {
    fn find(
        root: &'a mut Option<Root<DIM, GRID_EXPONENT>>,
        granularity: f64,
        bounds: Bounds<DIM>,
    ) -> Self {
        match root {
            None => {
                let root = root.insert(Root::new(granularity, bounds));
                InsertPoint {
                    embedding: &root.embedding,
                    node: &mut root.node,
                    node_level: root.coords.level,
                    cell: 0,
                    // The initial root node is the smallest node that the initial element can be
                    // stored in.
                    sieved: true,
                }
            }
            Some(root) => {
                root.ensure_origin_precedes(granularity, &bounds);
                let bounds = root.embedding.bounds_from_world(granularity, &bounds);
                let target = bounds.node_location::<GRID_EXPONENT>();
                let (node, level) = find_smallest_parent(&mut root.node, &mut root.coords, target);
                InsertPoint {
                    embedding: &root.embedding,
                    node,
                    node_level: level,
                    cell: grid_index_at_level::<DIM, GRID_EXPONENT>(bounds.min, level),
                    sieved: level == target.level,
                }
            }
        }
    }
}

fn balance_node<const DIM: usize, const GRID_EXPONENT: u32, T>(
    node: &mut Node<DIM, GRID_EXPONENT>,
    node_level: u32,
    embedding: &Embedding<DIM>,
    granularity: f64,
    elements: &mut Slab<Element<T>>,
    elements_per_cell: usize,
    mut get_bounds: impl FnMut(&T) -> Bounds<DIM>,
) {
    let mut split = |level, node: &mut Node<DIM, GRID_EXPONENT>| {
        if level == GRID_EXPONENT {
            // No further subdivision possible
            return;
        }
        if node
            .state
            .unsieved_elements()
            .iter()
            .all(|&n| n <= elements_per_cell)
        {
            // Node doesn't need to be split
            return;
        }
        let children = node.state.ensure_children();
        // Check every cell for unsieved elements to split
        for cell in &mut *node.grid {
            let mut next_elt = cell.first_element;
            let mut prev_elt = None;
            while let Some(element) = next_elt.get() {
                next_elt = elements[element].next;
                let bounds =
                    embedding.bounds_from_world(granularity, &get_bounds(&elements[element].value));
                let Some(child_node_idx) = bounds.index_in::<GRID_EXPONENT>(level - 1) else {
                    // Too large to move into children
                    prev_elt = Some(element);
                    continue;
                };
                let cell_idx = grid_index_at_level::<DIM, GRID_EXPONENT>(bounds.min, level - 1);
                let target_level = bounds.level::<GRID_EXPONENT>();
                // Link into child
                link(
                    elements,
                    &mut children[child_node_idx],
                    cell_idx,
                    element,
                    target_level == level - 1,
                );
                // Unlink from `node`
                let prev_link = match prev_elt {
                    None => &mut cell.first_element,
                    Some(x) => &mut elements[x].next,
                };
                *prev_link = next_elt;
            }
        }
    };

    // See comment on `DepthFirstTraversal::queue`
    let mut stack = ArrayVec::<(u32, &mut [Node<DIM, GRID_EXPONENT>]), MAX_DEPTH>::new();
    split(node_level, node);
    if let Some(children) = node.state.children_mut() {
        stack.push((node_level, children));
    }
    while let Some((level, children)) = stack.pop() {
        let level = level - 1;
        for child in children.iter_mut() {
            // Balance elements in `child`
            split(level, child);

            // Queue grandchildren for balancing
            if let Some(grandchildren) = child.state.children_mut() {
                stack.push((level, grandchildren));
            }
        }
    }
}

#[derive(Debug)]
struct Root<const DIM: usize, const GRID_EXPONENT: u32> {
    embedding: Embedding<DIM>,
    coords: CellCoords<DIM>,
    node: Node<DIM, GRID_EXPONENT>,
}

impl<const DIM: usize, const GRID_EXPONENT: u32> Root<DIM, GRID_EXPONENT> {
    fn new(granularity: f64, first_element: Bounds<DIM>) -> Self {
        let embedding = Embedding {
            origin: first_element.min,
        };
        let extent = embedding
            .bounds_from_world(granularity, &first_element)
            .extents()
            .into_iter()
            .max();
        let coords = CellCoords {
            min: [0; DIM],
            level: extent.map_or(0, level_for_extent) + GRID_EXPONENT,
        };
        Root {
            embedding,
            coords,
            node: Node::default(),
        }
    }

    fn world_bounds(&self, granularity: f64) -> Bounds<DIM> {
        self.embedding
            .world_bounds_from_tree(granularity, &self.coords.bounds())
    }

    fn ensure_origin_precedes(&mut self, granularity: f64, bounds: &Bounds<DIM>) {
        let current = self.world_bounds(granularity);
        if bounds.min.iter().zip(&current.min).any(|(x, y)| x < y) {
            // `bounds` falls below the area currently covered by the tree. Shift the origin
            // by a multiple of the root node size to encompass it, and shift the root node
            // in the opposite direction so we don't have to reindex.

            let root_extent = cell_extent(self.coords.level);
            let world_root_extent = granularity * root_extent as f64;
            let offset: [u64; DIM] = array::from_fn(|i| {
                let min = current.min[i].min(bounds.min[i]);
                // Nonnegative
                let distance = self.embedding.origin[i] - min;
                // How far we need to shift the origin in this dimension, in multiples of extent
                let offset = distance / world_root_extent;
                // `offset.ceil() as u64` would be a little cleaner, but isn't in core.
                if offset == 0.0 {
                    0
                } else {
                    offset as u64 + 1
                }
            });
            // Move the origin downwards to encompass the new minimum bound...
            self.embedding.origin =
                array::from_fn(|i| self.embedding.origin[i] - offset[i] as f64 * world_root_extent);
            // ...and move the root node upwards to compensate.
            self.coords.min = array::from_fn(|i| self.coords.min[i] + offset[i] * root_extent);
        }
    }
}

/// Look up the smallest existing parent of `target`, uprooting the tree if necessary
fn find_smallest_parent<'a, const DIM: usize, const GRID_EXPONENT: u32>(
    root: &'a mut Node<DIM, GRID_EXPONENT>,
    root_coords: &mut CellCoords<DIM>,
    target: CellCoords<DIM>,
) -> (&'a mut Node<DIM, GRID_EXPONENT>, u32) {
    let ancestor = root_coords.smallest_common_ancestor(&target);
    if ancestor == *root_coords {
        return find_smallest_existing_parent(root_coords.level, root, target);
    }
    // Create new root that encloses both old root and target
    let old_root = mem::take(root);
    let old_root_coords = mem::replace(root_coords, ancestor);

    // Reattach the old root under the new root
    let mut current = &mut *root;
    let mut current_level = root_coords.level;
    while current_level > old_root_coords.level {
        let children = current.state.ensure_children();
        current_level -= 1;
        let index = child_index_at_level::<DIM>(old_root_coords.min, current_level);
        current = &mut children[index];
    }
    *current = old_root;

    let (level, node) = if target.level < ancestor.level {
        // Return the child of the new root that encloses `target`
        let index = child_index_at_level::<DIM>(target.min, root_coords.level - 1);
        (
            root_coords.level - 1,
            &mut root.state.children_mut().unwrap()[index],
        )
    } else {
        // Ancestor is target
        (root_coords.level, root)
    };
    (node, level)
}

fn find_smallest_existing_parent<'a, const DIM: usize, const GRID_EXPONENT: u32>(
    start_level: u32,
    start_node: &'a mut Node<DIM, GRID_EXPONENT>,
    target: CellCoords<DIM>,
) -> (&'a mut Node<DIM, GRID_EXPONENT>, u32) {
    let mut current = start_node;
    let mut current_level = start_level;
    {
        while let Some(children) = current.state.children_mut() {
            if current_level == target.level {
                break;
            }
            current_level -= 1;
            let index = child_index_at_level::<DIM>(target.min, current_level);
            // Hack around borrowck limitation
            unsafe {
                current = mem::transmute::<
                    &mut Node<DIM, GRID_EXPONENT>,
                    &'a mut Node<DIM, GRID_EXPONENT>,
                >(&mut children[index]);
            }
        }
    }
    (current, current_level)
}

/// Add `element` to `cell`
fn link<const DIM: usize, const GRID_EXPONENT: u32, T>(
    elements: &mut Slab<Element<T>>,
    node: &mut Node<DIM, GRID_EXPONENT>,
    cell_idx: usize,
    element: usize,
    sieved: bool,
) {
    let cell = &mut node.grid[cell_idx];
    let prev = mem::replace(&mut cell.first_element, Some(element).into());
    elements[element].next = prev;
    if !sieved {
        node.state.add_unsieved(cell_idx);
    }
}

/// Remove `element` from `cell`
fn unlink<const DIM: usize, const GRID_EXPONENT: u32, T>(
    elements: &mut Slab<Element<T>>,
    node: &mut Node<DIM, GRID_EXPONENT>,
    cell_idx: usize,
    element: usize,
    sieved: bool,
) {
    let successor = elements[element].next;
    let cell = &mut node.grid[cell_idx];
    let mut link = &mut cell.first_element;
    loop {
        let i = link.get().expect("element missing from node list");
        if i == element {
            *link = successor;
            break;
        }
        link = &mut elements[i].next;
    }
    if !sieved {
        node.state.remove_unsieved(cell_idx);
    }
}

/// Mapping between tree coordinates and world coordinates
#[derive(Debug)]
struct Embedding<const DIM: usize> {
    /// Lower bound of the tree in world space
    origin: [f64; DIM],
}

impl<const DIM: usize> Embedding<DIM> {
    /// Compute the location of the level-0 node that contains `world`
    fn tree_from_world(&self, granularity: f64, world: &[f64; DIM]) -> [u64; DIM] {
        array::from_fn(|i| ((world[i] - self.origin[i]) / granularity) as u64)
    }

    /// Compute the tree bounds that contain `world`
    fn bounds_from_world(&self, granularity: f64, world: &Bounds<DIM>) -> TreeBounds<DIM> {
        TreeBounds {
            min: self.tree_from_world(granularity, &world.min),
            max: self.tree_from_world(granularity, &world.max),
        }
    }

    /// Compute the lower bound of the world coordinates of the level-0 node at `tree`
    fn world_from_tree(&self, granularity: f64, tree: &[u64; DIM]) -> [f64; DIM] {
        array::from_fn(|i| tree[i] as f64 * granularity + self.origin[i])
    }

    fn world_bounds_from_tree(&self, granularity: f64, tree: &TreeBounds<DIM>) -> Bounds<DIM> {
        Bounds {
            min: self.world_from_tree(granularity, &tree.min),
            max: self.world_from_tree(granularity, &tree.max.map(|x| x + 1)),
        }
    }
}

/// Compute the index of the node at `level` containing `point` in its parent's child array
///
/// Equivalent to `CellCoords::from_point(point, level).index_in_parent()`
fn child_index_at_level<const DIM: usize>(point: [u64; DIM], level: u32) -> usize {
    let extent = cell_extent(level);
    let local_coords = point.map(|x| (x / extent) % SUBDIV as u64);
    index_from_local_coords(&local_coords, SUBDIV.into())
}

/// Index of the cell containing `point` in the grid of a node at `level`
fn grid_index_at_level<const DIM: usize, const GRID_EXPONENT: u32>(
    point: [u64; DIM],
    level: u32,
) -> usize {
    let extent = cell_extent(level - GRID_EXPONENT);
    let local_coords = point.map(|x| (x / extent) % grid_size::<GRID_EXPONENT>() as u64);
    index_from_local_coords(&local_coords, grid_size::<GRID_EXPONENT>() as u64)
}

/// Compute the lowest level a value with maximum bounding box edge length `extent` may occupy
fn level_for_extent(extent: u64) -> u32 {
    assert_eq!(SUBDIV, 2);
    extent.ilog2()
}

/// An axis-aligned bounding box in world space
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Bounds<const DIM: usize> {
    /// Smallest point inside the box
    pub min: [f64; DIM],
    /// Largest point inside the box
    pub max: [f64; DIM],
}

impl<const DIM: usize> Bounds<DIM> {
    /// Construct a rectangle covering a single coordinate
    #[cfg(test)]
    const fn point(p: [f64; DIM]) -> Self {
        Self { min: p, max: p }
    }
}

struct Node<const DIM: usize, const GRID_EXPONENT: u32> {
    state: NodeState<DIM, GRID_EXPONENT>,
    // This should become `[Node<DIM, GRID_EXPONENT>; SUBDIV.pow(GRID_EXPONENT).pow(DIM)]` as soon as Rust permits that
    grid: Box<[Cell]>,
}

impl<const DIM: usize, const GRID_EXPONENT: u32> fmt::Debug for Node<DIM, GRID_EXPONENT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("state", &self.state)
            .field("elements", &GridElements::<DIM, GRID_EXPONENT>(&*self.grid))
            .finish()
    }
}

struct GridElements<'a, const DIM: usize, const GRID_EXPONENT: u32>(&'a [Cell]);

impl<'a, const DIM: usize, const GRID_EXPONENT: u32> fmt::Debug
    for GridElements<'a, DIM, GRID_EXPONENT>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.0.iter().enumerate().filter_map(|(idx, cell)| {
                Some((
                    index_coords::<DIM, GRID_EXPONENT>(idx),
                    cell.first_element.get()?,
                ))
            }))
            .finish()
    }
}

/// Compute the coordinates of an index in a dense `DIM`-dimensional grid with edges of length
/// `SUBDIV.pow(GRID_EXPONENT)`
fn index_coords<const DIM: usize, const GRID_EXPONENT: u32>(index: usize) -> [u64; DIM] {
    let range = (SUBDIV as u64).pow(GRID_EXPONENT);
    array::from_fn(|i| {
        let unit = range.pow(i as u32);
        (index as u64 / unit) % range
    })
}

#[derive(Debug)]
enum NodeState<const DIM: usize, const GRID_EXPONENT: u32> {
    /// Has children, and contains only values too large to be stored at a lower level
    Internal {
        // This should become `Box<[Node<DIM, GRID_EXPONENT>; SUBDIV.pow(DIM)]>` as soon as Rust permits that
        children: Box<[Node<DIM, GRID_EXPONENT>]>,
    },
    /// No children
    Leaf {
        /// Number of values that could be moved into a child node, per cell
        unsieved_elements: Box<[usize]>,
    },
}

impl<const DIM: usize, const GRID_EXPONENT: u32> NodeState<DIM, GRID_EXPONENT> {
    fn ensure_children(&mut self) -> &mut [Node<DIM, GRID_EXPONENT>] {
        match self {
            NodeState::Internal { children } => children,
            NodeState::Leaf { .. } => {
                *self = NodeState::Internal {
                    children: (0..SUBDIV.pow(DIM as u32) as usize)
                        .map(|_| Node::default())
                        .collect(),
                };
                match self {
                    NodeState::Internal { children } => children,
                    _ => unreachable!(),
                }
            }
        }
    }

    fn unsieved_elements(&self) -> &[usize] {
        match self {
            NodeState::Internal { .. } => &[],
            NodeState::Leaf { unsieved_elements } => unsieved_elements,
        }
    }

    fn add_unsieved(&mut self, cell: usize) {
        match *self {
            NodeState::Internal { .. } => unreachable!("adding unsieved element to internal node"),
            NodeState::Leaf {
                ref mut unsieved_elements,
            } => unsieved_elements[cell] += 1,
        }
    }

    fn remove_unsieved(&mut self, cell: usize) {
        match *self {
            NodeState::Internal { .. } => {
                unreachable!("removing unsieved element from internal node")
            }
            NodeState::Leaf {
                ref mut unsieved_elements,
            } => unsieved_elements[cell] -= 1,
        }
    }

    fn children(&self) -> Option<&[Node<DIM, GRID_EXPONENT>]> {
        match *self {
            NodeState::Internal { ref children } => Some(children),
            NodeState::Leaf { .. } => None,
        }
    }

    fn children_mut(&mut self) -> Option<&mut [Node<DIM, GRID_EXPONENT>]> {
        match *self {
            NodeState::Internal { ref mut children } => Some(children),
            NodeState::Leaf { .. } => None,
        }
    }
}

impl<const DIM: usize, const GRID_EXPONENT: u32> Default for Node<DIM, GRID_EXPONENT> {
    fn default() -> Self {
        Self {
            state: NodeState::Leaf {
                unsieved_elements: (0..SUBDIV.pow(GRID_EXPONENT).pow(DIM as u32))
                    .map(|_| 0)
                    .collect(),
            },
            grid: (0..SUBDIV.pow(GRID_EXPONENT).pow(DIM as u32))
                .map(|_| Cell::default())
                .collect(),
        }
    }
}

#[derive(Debug, Default)]
struct Cell {
    first_element: MaybeIndex,
}

#[derive(Debug)]
struct Element<T> {
    value: T,
    next: MaybeIndex,
}

/// Index of coordinates in a cuboidal `grid_size.pow(DIM)` grid
fn index_from_local_coords<const DIM: usize>(local_coords: &[u64; DIM], grid_size: u64) -> usize {
    local_coords
        .iter()
        .enumerate()
        .map(|(i, &x)| (x * grid_size.pow(i as u32)) as usize)
        .sum()
}

/// A tree of branching factor 2 covering the entire range of u64 can be at most this deep before
/// nodes can no longer be subdivided.
const MAX_DEPTH: usize = u64::MAX.ilog(2) as usize;

/// Each level subdivides space into this many parts along each dimension. Each node has
/// `SUBDIV.pow(DIM)` children.
const SUBDIV: u32 = 2;

const fn grid_size<const GRID_EXPONENT: u32>() -> usize {
    SUBDIV.pow(GRID_EXPONENT) as usize
}

#[derive(Copy, Clone)]
struct MaybeIndex(usize);

impl MaybeIndex {
    fn new(i: Option<usize>) -> Self {
        Self(i.unwrap_or(usize::MAX))
    }

    fn get(self) -> Option<usize> {
        match self.0 == usize::MAX {
            true => None,
            false => Some(self.0),
        }
    }
}

impl From<Option<usize>> for MaybeIndex {
    fn from(x: Option<usize>) -> Self {
        Self::new(x)
    }
}

impl Default for MaybeIndex {
    fn default() -> Self {
        Self::new(None)
    }
}

impl fmt::Debug for MaybeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 == usize::MAX {
            false => f.debug_tuple("Some").field(&self.0).finish(),
            true => f.debug_tuple("None").finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use traversal::ElementIter;

    #[test]
    fn common_ancestors() {
        type Node1D = CellCoords<1>;

        let min = Node1D::from_point([0], 0);
        assert_eq!(min.smallest_common_ancestor(&min), min);
        assert_eq!(
            Node1D::from_point([1], 0).smallest_common_ancestor(&min),
            Node1D::from_point([0], 1)
        );
        assert_eq!(
            Node1D::from_point([1], 0).smallest_common_ancestor(&min),
            Node1D::from_point([0], 1)
        );
        assert_eq!(
            Node1D::from_point([2], 0).smallest_common_ancestor(&min),
            Node1D::from_point([0], 2)
        );
        assert_eq!(
            Node1D::from_point([3], 0).smallest_common_ancestor(&min),
            Node1D::from_point([0], 2)
        );
    }

    #[test]
    fn index_in_parent_leaf() {
        let origin = SUBDIV * 7;
        for y in 0..SUBDIV {
            for x in 0..SUBDIV {
                assert_eq!(
                    CellCoords::<2>::from_point([(origin + x).into(), (origin + y).into()], 0)
                        .index_in_parent(),
                    (y * SUBDIV + x) as usize
                );
            }
        }
    }

    #[test]
    fn index_in_parent_mid() {
        assert_eq!(CellCoords::<1>::from_point([5], 2).index_in_parent(), 1);
    }

    #[test]
    fn balance() {
        let mut t = SieveTree::<2, 2, Bounds<2>>::new();
        for y in -5..5 {
            for x in -5..5 {
                let b = Bounds::point([x as f64, y as f64]);
                t.insert(b, b);
                validate(&t);
            }
        }
        t.balance(1, |&x| x);
        validate(&t);
    }

    #[test]
    fn smoke() {
        let mut t = SieveTree::<2, 2, Bounds<2>>::new();
        let b = Bounds {
            min: [4.0, 4.0],
            max: [107.0, 107.0],
        };
        t.insert(b, b);
        assert_eq!(
            t.intersections(Bounds {
                min: [0.0, 0.0],
                max: [10.0, 10.0]
            })
            .count(),
            1
        );
        assert_eq!(
            t.intersections(Bounds {
                min: [10.0, 20.0],
                max: [30.0, 40.0]
            })
            .count(),
            1
        );

        assert_eq!(
            t.intersections(Bounds {
                min: [1000.0, 1000.0],
                max: [1001.0, 1001.0],
            })
            .count(),
            0
        );
    }

    /// Assert that each element is the right cell
    #[track_caller]
    fn validate<const DIM: usize, const GRID_EXPONENT: u32>(
        tree: &SieveTree<DIM, GRID_EXPONENT, Bounds<DIM>>,
    ) {
        let mut stack = alloc::vec::Vec::new();
        if let Some(root) = &tree.root {
            stack.push((root.coords, &root.node));
        }
        let mut total_elements = 0;
        while let Some((coords, node)) = stack.pop() {
            if let Some(children) = node.state.children() {
                let child_extent = cell_extent(coords.level - 1);
                for (local, child) in grid::<DIM>(SUBDIV as u64).zip(children.as_ref()) {
                    let child_coords = CellCoords {
                        min: array::from_fn(|i| coords.min[i] + local[i] * child_extent),
                        level: coords.level - 1,
                    };
                    stack.push((child_coords, child));
                }
            }

            for (local, cell) in
                grid::<DIM>(grid_size::<GRID_EXPONENT>() as u64).zip(node.grid.as_ref())
            {
                let extent = cell_extent(coords.level - GRID_EXPONENT);
                let parent_extent = extent * SUBDIV as u64;
                let cell_coords = CellCoords::<DIM> {
                    min: array::from_fn(|i| coords.min[i] + local[i] * extent),
                    level: coords.level - GRID_EXPONENT,
                };
                let cell_bounds = cell_coords.bounds();
                let mut iter = ElementIter::new(cell.first_element.get());
                while let Some(elt) = iter.next(&tree.elements) {
                    total_elements += 1;
                    let elt_bounds = tree
                        .root
                        .as_ref()
                        .unwrap()
                        .embedding
                        .bounds_from_world(tree.granularity, &tree.elements[elt].value);
                    assert!(
                        cell_bounds.contains(&elt_bounds.min),
                        "element {} origin {:?} outside of cell {} with bounds {}",
                        elt,
                        elt_bounds.min,
                        cell_coords,
                        cell_bounds
                    );
                    let max_extent = elt_bounds.extents().into_iter().max().unwrap_or(0);
                    assert!(
                        max_extent < parent_extent,
                        "element {} extent {} too large for cell extent {}",
                        elt,
                        max_extent,
                        extent
                    );
                    if node.state.children().is_some() {
                        assert!(
                            max_extent >= extent,
                            "element {} extent {} too small for non-leaf node {} with cell extent {}",
                            elt,
                            max_extent,
                            coords,
                            extent
                        );
                    }
                }
            }
        }
        assert_eq!(
            total_elements,
            tree.elements.len(),
            "{} elements leaked",
            tree.elements.len() - total_elements
        );
    }

    /// Iterator over all relative coordinates within a cuboid grid
    fn grid<const DIM: usize>(size: u64) -> impl Iterator<Item = [u64; DIM]> {
        let mut cursor = [0u64; DIM];
        core::iter::from_fn(move || {
            if *cursor.last()? == size {
                return None;
            }
            let result = cursor;
            cursor[0] += 1;
            for d in 1..DIM {
                if cursor[d - 1] < size {
                    break;
                }
                cursor[d - 1] = 0;
                cursor[d] += 1;
            }
            Some(result)
        })
    }

    #[test]
    fn trivial_remove() {
        let mut tree = SieveTree::<1, 2, Bounds<1>>::new();
        let bounds = Bounds::point([0.]);
        let id = tree.insert(bounds, bounds);
        assert_eq!(tree.remove(id, bounds), bounds);
    }

    #[test]
    fn index_coords_sanity() {
        // Indices in a 2x2x2 cube
        assert_eq!(index_coords::<3, 1>(0), [0, 0, 0]);
        assert_eq!(index_coords::<3, 1>(1), [1, 0, 0]);
        assert_eq!(index_coords::<3, 1>(2), [0, 1, 0]);
        assert_eq!(index_coords::<3, 1>(3), [1, 1, 0]);
        assert_eq!(index_coords::<3, 1>(4), [0, 0, 1]);
        assert_eq!(index_coords::<3, 1>(5), [1, 0, 1]);
        assert_eq!(index_coords::<3, 1>(6), [0, 1, 1]);
        assert_eq!(index_coords::<3, 1>(7), [1, 1, 1]);
    }

    #[test]
    fn regression1() {
        let mut t = SieveTree::<2, 2, Bounds<2>>::new();
        let b1 = Bounds {
            min: [30., 30.],
            max: [31., 31.],
        };
        let b2 = Bounds {
            min: [50., 50.],
            max: [51., 51.],
        };
        t.insert(b1, b1);
        t.insert(b2, b2);
        assert_eq!(
            t.intersections(Bounds {
                min: [0.0, 0.0],
                max: [100., 100.]
            })
            .count(),
            2
        );
    }

    #[test]
    fn regression2() {
        let mut t = SieveTree::<1, 2, Bounds<1>>::with_granularity(1.);
        let b1 = Bounds {
            min: [5.],
            max: [10.],
        };
        let b2 = Bounds {
            min: [20.],
            max: [25.],
        };
        t.insert(b1, b1);
        t.insert(b2, b2);

        let intersections = t
            .intersections(Bounds {
                min: [15.],
                max: [45.],
            })
            .map(|(_, bounds)| *bounds)
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(
            intersections,
            alloc::vec![Bounds {
                min: [20.],
                max: [25.],
            }],
        );
    }
}
