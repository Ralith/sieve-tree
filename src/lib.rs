#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use core::{array, fmt, mem};

use arrayvec::ArrayVec;
use slab::Slab;

/// A `DIM`-dimensional spatial search tree
///
/// Each tree node owns a grid of `2.pow(GRID_EXPONENT).pow(DIM)` cells. Increasing `GRID_EXPONENT`
/// makes the tree less sparse, which accelerates random access by reducing indirection in exchange
/// for exponentially increased memory requirements and balance work.
#[derive(Debug)]
pub struct SieveTree<const DIM: usize, const GRID_EXPONENT: u32, T> {
    /// Length of a level 0 node edge in world space
    scale: f64,
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
    pub fn with_scale(scale: f64) -> Self {
        Self {
            scale,
            ..Self::default()
        }
    }

    /// Insert `value` into the best existing location in the tree, returning an ID that can be used
    /// to access it directly
    pub fn insert(&mut self, bounds: Bounds<DIM>, value: T) -> usize {
        let id = self.elements.insert(Element { value, next: None });

        let cell = match &mut self.root {
            None => {
                let embedding = Embedding { origin: bounds.min };
                let extent = embedding
                    .bounds_from_world(self.scale, &bounds)
                    .extents()
                    .into_iter()
                    .max();
                let coords = CellCoords {
                    min: [0; DIM],
                    level: extent.map_or(0, level_for_extent) + GRID_EXPONENT,
                };
                let node = &mut self
                    .root
                    .insert(Root {
                        embedding,
                        coords,
                        node: Node::default(),
                    })
                    .node;
                &mut node.grid[0]
            }
            Some(root) => {
                let current = root.world_bounds(self.scale);
                if bounds.min.iter().zip(&current.min).any(|(x, y)| x < y) {
                    // `bounds` falls below the area currently covered by the tree. Shift the origin
                    // by a multiple of the root node size to encompass it, and shift the root node
                    // in the opposite direction so we don't have to reindex.

                    let root_extent = cell_extent(root.coords.level);
                    let world_root_extent = self.scale * root_extent as f64;
                    let offset: [u64; DIM] = array::from_fn(|i| {
                        let min = current.min[i].min(bounds.min[i]);
                        // Nonnegative
                        let distance = root.embedding.origin[i] - min;
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
                    root.embedding.origin = array::from_fn(|i| {
                        root.embedding.origin[i] - offset[i] as f64 * world_root_extent
                    });
                    // ...and move the root node upwards to compensate.
                    root.coords.min =
                        array::from_fn(|i| root.coords.min[i] + offset[i] * root_extent);
                }

                let target = root
                    .embedding
                    .bounds_from_world(self.scale, &bounds)
                    .location();
                find_smallest_parent(root, target)
            }
        };
        link(&mut self.elements, cell, id);

        id
    }

    /// Recursively split cells with more than `elements_per_cell` elements
    ///
    /// Call after large numbers of `insert`s to maintain consistent search performance.
    pub fn balance(
        &mut self,
        elements_per_cell: usize,
        mut get_bounds: impl FnMut(&T) -> Bounds<DIM>,
    ) {
        let Some(root) = &mut self.root else {
            return;
        };
        let mut split = |level, node: &mut Node<DIM, GRID_EXPONENT>| {
            if level == GRID_EXPONENT {
                // No further subdivision possible
                return;
            }
            let mut any_fresh_split = false;
            for cell in &mut *node.grid {
                // Split a cell if it's too large. The first time we allocate children, we visit all
                // (non-sieved) elements to ensure small values are always in the smallest possible
                // cell, and don't float around high up in the tree indefinitely.
                if cell.elements <= elements_per_cell && !any_fresh_split {
                    continue;
                }
                let (fresh_split, children) = ensure_children(&mut node.children);
                any_fresh_split |= fresh_split;
                let mut next_elt = cell.first_element;
                let mut prev_elt = None;
                while let Some(element) = next_elt {
                    next_elt = self.elements[element].next;
                    let bounds = root
                        .embedding
                        .bounds_from_world(self.scale, &get_bounds(&self.elements[element].value));
                    let Some(child_node_idx) = bounds.index_in::<GRID_EXPONENT>(level - 1) else {
                        // Too large to move into children
                        prev_elt = Some(element);
                        continue;
                    };
                    let cell_idx = grid_index_at_level::<DIM, GRID_EXPONENT>(bounds.min, level - 1);
                    // Link into child
                    link(
                        &mut self.elements,
                        &mut children[child_node_idx].grid[cell_idx],
                        element,
                    );
                    // Unlink from `node`
                    let prev_link = match prev_elt {
                        None => &mut cell.first_element,
                        Some(x) => &mut self.elements[x].next,
                    };
                    *prev_link = next_elt;
                    cell.elements -= 1;
                }
            }
        };

        // See comment on `DepthFirstTraversal::queue`
        let mut stack = ArrayVec::<(u32, &mut [Node<DIM, GRID_EXPONENT>]), MAX_DEPTH>::new();
        split(root.coords.level, &mut root.node);
        if let Some(children) = root.node.children.as_mut() {
            stack.push((root.coords.level, children));
        }
        while let Some((level, children)) = stack.pop() {
            let level = level - 1;
            for child in children.iter_mut() {
                // Balance elements in `child`
                split(level, child);

                // Queue grandchildren for balancing
                if let Some(grandchildren) = child.children.as_mut() {
                    stack.push((level, grandchildren));
                }
            }
        }
    }

    /// Remove the value associated with `id`
    pub fn remove(&mut self, bounds: Bounds<DIM>, id: usize) -> T {
        let elt = self.elements.remove(id);
        let root = self.root.as_mut().unwrap();
        let bounds = root.embedding.bounds_from_world(self.scale, &bounds);
        let target = bounds.location();
        // A value is guaranteed to be stored in the smallest existing node permitted for it, because:
        // - `insert` only introduces nodes that are siblings of or larger than the root
        // - `balance` always moves all possible elements into newly created child nodes
        let cell = find_smallest_existing_parent(root, target);
        unlink(&mut self.elements, cell, id);
        elt.value
    }

    pub fn bounds(&self) -> Option<Bounds<DIM>> {
        self.root.as_ref().map(|root| root.world_bounds(self.scale))
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
            .map(|x| x.embedding.bounds_from_world(self.scale, &bounds))
            .unwrap_or(TreeBounds {
                min: [0; DIM],
                max: [0; DIM],
            });
        let mut out = Intersections {
            bounds,
            elements: &self.elements,
            traversal: DepthFirstTraversal::default(),
            next_child: CellsWithin::default(),
            children: &[],
            next_cell: CellsWithin::default(),
            next_element: ElementIter::default(),
            grid: &[],
        };
        if let Some(ref root) = self.root {
            if bounds.intersects(&root.coords.bounds()) {
                out.traversal.push(root.coords, &root.node);
                out.grid = &root.node.grid;
                out.next_cell = root.coords.cells_overlapping::<GRID_EXPONENT>(&bounds);
            }
        }
        out
    }
}

impl<const DIM: usize, const GRID_EXPONENT: u32, T> Default for SieveTree<DIM, GRID_EXPONENT, T> {
    fn default() -> Self {
        Self {
            // 1cm if world space is in meters. Big enough for the solar system.
            scale: 0.01,
            root: None,
            elements: Slab::default(),
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
    fn world_bounds(&self, scale: f64) -> Bounds<DIM> {
        self.embedding
            .world_bounds_from_tree(scale, &self.coords.bounds())
    }
}

/// Look up the smallest existing parent of `target`, uprooting the tree if necessary
fn find_smallest_parent<const DIM: usize, const GRID_EXPONENT: u32>(
    root: &mut Root<DIM, GRID_EXPONENT>,
    target: CellCoords<DIM>,
) -> &mut Cell {
    let ancestor = root.coords.smallest_common_ancestor(&target);
    if ancestor == root.coords {
        return find_smallest_existing_parent(root, target);
    }
    // Create new root that encloses both old root and target
    let old_root = mem::take(&mut root.node);
    let old_root_coords = mem::replace(&mut root.coords, ancestor);

    // Reattach the old root under the new root
    let mut current = &mut root.node;
    let mut current_level = root.coords.level;
    while current_level > old_root_coords.level {
        let (_, children) = ensure_children(&mut current.children);
        current_level -= 1;
        let index = child_index_at_level::<DIM>(old_root_coords.min, current_level);
        current = &mut children[index];
    }
    *current = old_root;

    let (level, node) = if target.level < ancestor.level {
        // Return the child of the new root that encloses `target`
        let index = child_index_at_level::<DIM>(target.min, root.coords.level - 1);
        (
            root.coords.level - 1,
            &mut root.node.children.as_mut().unwrap()[index],
        )
    } else {
        // Ancestor is target
        (root.coords.level, &mut root.node)
    };
    let cell_index = grid_index_at_level::<DIM, GRID_EXPONENT>(target.min, level);
    &mut node.grid[cell_index]
}

fn find_smallest_existing_parent<'a, const DIM: usize, const GRID_EXPONENT: u32>(
    root: &'a mut Root<DIM, GRID_EXPONENT>,
    target: CellCoords<DIM>,
) -> &'a mut Cell {
    let mut current = &mut root.node;
    let mut current_level = root.coords.level;
    {
        while let Some(ref mut children) = current.children {
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
    let cell_index = grid_index_at_level::<DIM, GRID_EXPONENT>(target.min, current_level);
    &mut current.grid[cell_index]
}

/// Add `element` to `cell`
fn link<T>(elements: &mut Slab<Element<T>>, cell: &mut Cell, element: usize) {
    let prev = mem::replace(&mut cell.first_element, Some(element));
    elements[element].next = prev;
    cell.elements += 1;
}

/// Remove `element` from `cell`
fn unlink<T>(elements: &mut Slab<Element<T>>, cell: &mut Cell, element: usize) {
    let successor = elements[element].next;
    let mut link = &mut cell.first_element;
    loop {
        let i = link.expect("element missing from node list");
        if i == element {
            *link = successor;
            break;
        }
        link = &mut elements[i].next;
    }
    cell.elements -= 1;
}

/// Mapping between tree coordinates and world coordinates
#[derive(Debug)]
struct Embedding<const DIM: usize> {
    /// Lower bound of the tree in world space
    origin: [f64; DIM],
}

impl<const DIM: usize> Embedding<DIM> {
    /// Compute the location of the level-0 node that contains `world`
    fn tree_from_world(&self, scale: f64, world: &[f64; DIM]) -> [u64; DIM] {
        array::from_fn(|i| ((world[i] - self.origin[i]) / scale) as u64)
    }

    /// Compute the tree bounds that contain `world`
    fn bounds_from_world(&self, scale: f64, world: &Bounds<DIM>) -> TreeBounds<DIM> {
        TreeBounds {
            min: self.tree_from_world(scale, &world.min),
            max: self.tree_from_world(scale, &world.max),
        }
    }

    /// Compute the lower bound of the world coordinates of the level-0 node at `tree`
    fn world_from_tree(&self, scale: f64, tree: &[u64; DIM]) -> [f64; DIM] {
        array::from_fn(|i| tree[i] as f64 * scale + self.origin[i])
    }

    fn world_bounds_from_tree(&self, scale: f64, tree: &TreeBounds<DIM>) -> Bounds<DIM> {
        Bounds {
            min: self.world_from_tree(scale, &tree.min),
            max: self.world_from_tree(scale, &tree.max.map(|x| x + 1)),
        }
    }
}

/// Identifies a single cell, anywhere in tree space
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct CellCoords<const DIM: usize> {
    /// Point in this cell with the smallest coordinates on each dimension
    min: [u64; DIM],
    /// Exponent of `SUBDIV` which is the cell's extent in each dimension
    level: u32,
}

impl<const DIM: usize> CellCoords<DIM> {
    fn from_point(point: [u64; DIM], level: u32) -> Self {
        let extent = cell_extent(level);
        Self {
            min: point.map(|x| (x / extent) * extent),
            level,
        }
    }

    fn bounds(&self) -> TreeBounds<DIM> {
        let extent = cell_extent(self.level);
        TreeBounds {
            min: self.min,
            max: self.min.map(|x| x + extent - 1),
        }
    }

    fn parent(&self) -> Self {
        Self::from_point(self.min, self.level + 1)
    }

    fn index_in_parent(&self) -> usize {
        let extent = cell_extent(self.level);
        let local_coords = self.min.map(|x| (x / extent) % SUBDIV as u64);
        index_from_local_coords(&local_coords, SUBDIV.into())
    }

    fn index_in_grid<const GRID_EXPONENT: u32>(&self) -> usize {
        let extent = cell_extent(self.level);
        let local_coords = self
            .min
            .map(|x| (x / extent) % grid_size::<GRID_EXPONENT>() as u64);
        index_from_local_coords(&local_coords, grid_size::<GRID_EXPONENT>() as u64)
    }

    /// Iterator over child nodes in a node that might overlap with `bounds`
    fn children_overlapping(&self, bounds: &TreeBounds<DIM>) -> Option<CellsWithin<DIM>> {
        let level = self.level.checked_sub(1)?;
        let extent = cell_extent(level);
        // Expand search by one unit towards the origin to allow for edge-crossing
        let bounds = bounds.relax(extent);
        let mut range = self.bounds().intersection(&bounds);
        // Clamp lower bound to node boundaries
        range.min = range.min.map(|x| x - x % extent);
        Some(CellsWithin::new(range, level))
    }

    /// Iterator over grid cells in a node that might overlap with `bounds`
    fn cells_overlapping<const GRID_EXPONENT: u32>(
        &self,
        bounds: &TreeBounds<DIM>,
    ) -> CellsWithin<DIM> {
        let level = self.level - GRID_EXPONENT;
        let extent = cell_extent(level);
        // Expand search by one unit towards the origin to allow for edge-crossing
        let bounds = bounds.relax(extent);
        let mut range = self.bounds().intersection(&bounds);
        // Clamp lower bound to node boundaries
        range.min = range.min.map(|x| x - x % extent);
        CellsWithin::new(range, level)
    }

    fn smallest_common_ancestor(&self, other: &Self) -> Self {
        let mut a = *self;
        let mut b = *other;
        // Ensure `a` is the largest node
        if a.level < b.level {
            mem::swap(&mut a, &mut b);
        }
        // Find the parent of the smaller node which is on the larger node's level
        while a.level > b.level {
            b = b.parent();
        }
        // Find the common parent
        while a.min != b.min {
            a = a.parent();
            b = b.parent();
        }
        a
    }
}

impl<const DIM: usize> fmt::Display for CellCoords<DIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extent = cell_extent(self.level);
        write!(f, "{}:[", self.level)?;
        for (i, x) in self.min.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", x / extent)?;
        }
        write!(f, "]")
    }
}

/// Iterator over all nodes at a certain level with a min point inside a [`TreeBounds`]
#[derive(Debug)]
struct CellsWithin<const DIM: usize> {
    range: TreeBounds<DIM>,
    cursor: [u64; DIM],
    level: u32,
}

impl<const DIM: usize> CellsWithin<DIM> {
    fn new(range: TreeBounds<DIM>, level: u32) -> Self {
        Self {
            range,
            cursor: range.min,
            level,
        }
    }
}

impl<const DIM: usize> Iterator for CellsWithin<DIM> {
    type Item = CellCoords<DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.last().unwrap() > self.range.max.last().unwrap() {
            return None;
        }
        let result = CellCoords {
            level: self.level,
            min: self.cursor,
        };
        let extent = cell_extent(self.level);
        self.cursor[0] += extent;
        for i in 1..DIM {
            if self.cursor[i - 1] <= self.range.max[i - 1] {
                break;
            }
            self.cursor[i - 1] = self.range.min[i - 1];
            self.cursor[i] += extent;
        }
        Some(result)
    }
}

impl<const DIM: usize> Default for CellsWithin<DIM> {
    /// Construct the empty iterator
    fn default() -> Self {
        Self {
            range: TreeBounds {
                min: [0; DIM],
                max: [0; DIM],
            },
            cursor: [1; DIM],
            level: 0,
        }
    }
}

const fn cell_extent(level: u32) -> u64 {
    (SUBDIV as u64).saturating_pow(level)
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
    local_coords
        .into_iter()
        .enumerate()
        .map(|(i, x)| x as usize * grid_size::<GRID_EXPONENT>().pow(i as u32))
        .sum()
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

/// An axis-aligned bounding box in tree space
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct TreeBounds<const DIM: usize> {
    /// Smallest point inside the box
    pub min: [u64; DIM],
    /// Largest point inside the box
    pub max: [u64; DIM],
}

impl<const DIM: usize> TreeBounds<DIM> {
    /// Number of points inside the rectangle on each axis
    fn extents(&self) -> [u64; DIM] {
        array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }

    /// Find the smallest node cell a value with these bounds could be stored in, i.e. the largest
    /// level with cells smaller than this `Bounds`'s extents on any dimension
    fn location(&self) -> CellCoords<DIM> {
        let Some(extent) = self.extents().into_iter().max() else {
            // 0-dimensional case
            return CellCoords {
                level: 0,
                min: [0; DIM],
            };
        };

        let level = level_for_extent(extent);
        CellCoords::from_point(self.min, level)
    }

    /// Compute the index of the node at `level` containing this rect in its parent's child array,
    /// and the index in the selected node's grid, or `None` if the value must be stored at a higher
    /// level
    fn index_in<const GRID_EXPONENT: u32>(&self, level: u32) -> Option<usize> {
        let Some(max_extent) = self.extents().into_iter().max() else {
            // 0-dimensional case
            return Some(0);
        };
        let extent = cell_extent(level - GRID_EXPONENT);
        let node_extent = extent * SUBDIV.pow(GRID_EXPONENT) as u64;
        if max_extent > extent * u64::from(SUBDIV) {
            return None;
        }
        let local_coords = self.min.map(|x| (x / node_extent) % SUBDIV as u64);
        Some(index_from_local_coords(&local_coords, SUBDIV.into()))
    }

    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..DIM {
            if self.max[i] < other.min[i] || self.min[i] > other.max[i] {
                return false;
            }
        }
        true
    }

    fn intersection(&self, other: &Self) -> Self {
        Self {
            min: array::from_fn(|i| self.min[i].max(other.min[i])),
            max: array::from_fn(|i| self.max[i].min(other.max[i])),
        }
    }

    #[cfg(test)]
    fn contains(&self, point: &[u64; DIM]) -> bool {
        self.min
            .iter()
            .zip(&self.max)
            .zip(point)
            .all(|((min, max), x)| min <= x && x <= max)
    }

    /// Shift the minimum value towards the origin by `range`
    fn relax(&self, range: u64) -> Self {
        Self {
            min: self.min.map(|x| x.saturating_sub(range)),
            max: self.max,
        }
    }
}

impl<const DIM: usize> Default for TreeBounds<DIM> {
    fn default() -> Self {
        Self {
            min: [0; DIM],
            max: [0; DIM],
        }
    }
}

impl<const DIM: usize> fmt::Display for TreeBounds<DIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}-{:?}", self.min, self.max)
    }
}

#[derive(Default)]
struct DepthFirstTraversal<'a, const DIM: usize, const GRID_EXPONENT: u32> {
    // By tracking groups of children, rather than individual nodes, we can keep the stack size
    // to O(depth), whereas a naive depth-first traversal would require O(depth * branch^dim).
    queue: ArrayVec<(CellCoords<DIM>, &'a [Node<DIM, GRID_EXPONENT>]), MAX_DEPTH>,
}

impl<'a, const DIM: usize, const GRID_EXPONENT: u32> DepthFirstTraversal<'a, DIM, GRID_EXPONENT> {
    fn push(&mut self, coords: CellCoords<DIM>, node: &'a Node<DIM, GRID_EXPONENT>) {
        if let Some(ref children) = node.children {
            self.queue.push((coords, children));
        }
    }

    fn pop(&mut self) -> Option<(CellCoords<DIM>, &'a [Node<DIM, GRID_EXPONENT>])> {
        self.queue.pop()
    }
}

/// Iterator over nodes that might intersect with a [`Bounds`]
pub struct Intersections<'a, const DIM: usize, const GRID_EXPONENT: u32, T> {
    bounds: TreeBounds<DIM>,
    elements: &'a Slab<Element<T>>,
    traversal: DepthFirstTraversal<'a, DIM, GRID_EXPONENT>,
    next_child: CellsWithin<DIM>,
    children: &'a [Node<DIM, GRID_EXPONENT>],
    next_cell: CellsWithin<DIM>,
    grid: &'a [Cell],
    next_element: ElementIter,
}

impl<'a, const DIM: usize, const GRID_EXPONENT: u32, T> Iterator
    for Intersections<'a, DIM, GRID_EXPONENT, T>
{
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Traverse the current cell's elements
            if let Some(index) = self.next_element.next(self.elements) {
                let elt = &self.elements[index];
                return Some((index, &elt.value));
            }
            // If the current cell has no elements, find a new cell
            if let Some(coords) = self.next_cell.next() {
                self.next_element = ElementIter::new(
                    self.grid[coords.index_in_grid::<GRID_EXPONENT>()].first_element,
                );
                continue;
            }
            // If we're out of cells, get the next child node
            if let Some(coords) = self.next_child.next() {
                let index = coords.index_in_parent();
                let child = &self.children[index];
                self.traversal.push(coords, child);
                self.grid = &child.grid;
                self.next_cell = coords.cells_overlapping::<GRID_EXPONENT>(&self.bounds);
            }
            // If we're out of child nodes, get the next group from the queue
            let (coords, children) = self.traversal.pop()?;
            self.children = children;
            self.next_child = coords.children_overlapping(&self.bounds).unwrap();
        }
    }
}

#[derive(Debug)]
struct Node<const DIM: usize, const GRID_EXPONENT: u32> {
    // This should become `Box<[Node<DIM, GRID_EXPONENT>; SUBDIV.pow(DIM)]>` as soon as Rust permits that
    children: Option<Box<[Node<DIM, GRID_EXPONENT>]>>,
    // This should become `[Node<DIM, GRID_EXPONENT>; SUBDIV.pow(GRID_EXPONENT).pow(DIM)]` as soon as Rust permits that
    grid: Box<[Cell]>,
}

fn ensure_children<const DIM: usize, const GRID_EXPONENT: u32>(
    children: &mut Option<Box<[Node<DIM, GRID_EXPONENT>]>>,
) -> (bool, &mut [Node<DIM, GRID_EXPONENT>]) {
    match children {
        Some(x) => (false, x),
        None => (
            true,
            children.insert(
                (0..SUBDIV.pow(DIM as u32) as usize)
                    .map(|_| Node::default())
                    .collect(),
            ),
        ),
    }
}

impl<const DIM: usize, const GRID_EXPONENT: u32> Default for Node<DIM, GRID_EXPONENT> {
    fn default() -> Self {
        Self {
            children: None,
            grid: (0..SUBDIV.pow(GRID_EXPONENT).pow(DIM as u32))
                .map(|_| Cell::default())
                .collect(),
        }
    }
}

#[derive(Debug, Default)]
struct Cell {
    /// Number of elements associated directly with this cell
    // TODO: Count only unsieved elements
    elements: usize,
    first_element: Option<usize>,
}

#[derive(Debug)]
struct Element<T> {
    value: T,
    next: Option<usize>,
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

#[derive(Default)]
struct ElementIter {
    next: Option<usize>,
}

impl ElementIter {
    fn new(next: Option<usize>) -> Self {
        Self { next }
    }

    fn next<T>(&mut self, elements: &Slab<Element<T>>) -> Option<usize> {
        let i = self.next?;
        self.next = elements[i].next;
        Some(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                validate::<2, 2, Bounds<2>>(&t);
            }
        }
        t.balance(1, |&x| x);
        validate::<2, 2, Bounds<2>>(&t);
        let mut nonempty_cells = 0;
        for (coords, node) in nodes(&t) {
            for cell in &*node.grid {
                assert!(cell.elements <= 1, "too many elements at {}", coords);
                if cell.elements == 1 {
                    nonempty_cells += 1;
                }
            }
        }
        assert_eq!(nonempty_cells, 100);
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

    fn nodes<const DIM: usize, const GRID_EXPONENT: u32, T>(
        tree: &SieveTree<DIM, GRID_EXPONENT, T>,
    ) -> impl Iterator<Item = (u32, &'_ Node<DIM, GRID_EXPONENT>)> {
        let mut stack = alloc::vec::Vec::new();
        if let Some(root) = &tree.root {
            stack.push((root.coords.level, &root.node));
        }
        core::iter::from_fn(move || {
            let (level, node) = stack.pop()?;
            if let Some(children) = node.children.as_ref() {
                stack.extend(children.iter().map(|node| (level - 1, node)));
            }
            Some((level, node))
        })
    }

    /// Assert that each element is the right cell
    #[track_caller]
    fn validate<const DIM: usize, const GRID_EXPONENT: u32, T>(
        tree: &SieveTree<DIM, GRID_EXPONENT, Bounds<DIM>>,
    ) {
        let mut stack = alloc::vec::Vec::new();
        if let Some(root) = &tree.root {
            stack.push((root.coords, &root.node));
        }
        let mut total_elements = 0;
        while let Some((coords, node)) = stack.pop() {
            if let Some(ref children) = node.children {
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
                let mut iter = ElementIter::new(cell.first_element);
                while let Some(elt) = iter.next(&tree.elements) {
                    total_elements += 1;
                    let elt_bounds = tree
                        .root
                        .as_ref()
                        .unwrap()
                        .embedding
                        .bounds_from_world(tree.scale, &tree.elements[elt].value);
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
                    if node.children.is_some() {
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
}
