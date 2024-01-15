#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use core::{array, fmt, mem};

use arrayvec::ArrayVec;
use slab::Slab;

/// A `DIM`-dimensional spatial search tree
#[derive(Debug)]
pub struct SieveTree<const DIM: usize, T> {
    /// Length of a level 0 node edge in world space
    scale: f64,
    root: Option<Root<DIM>>,
    elements: Slab<Element<T>>,
}

impl<const DIM: usize, T> SieveTree<DIM, T> {
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

        let node = match &mut self.root {
            None => {
                let embedding = Embedding { origin: bounds.min };
                let extent = embedding
                    .bounds_from_world(self.scale, &bounds)
                    .extents()
                    .into_iter()
                    .max();
                let coords = CellCoords {
                    min: [0; DIM],
                    level: extent.map_or(0, level_for_extent),
                };
                &mut self
                    .root
                    .insert(Root {
                        embedding,
                        coords,
                        node: Node::default(),
                    })
                    .node
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
                find_smallest_parent::<DIM>(root, target)
            }
        };
        link(&mut self.elements, node, id);

        id
    }

    /// Recursively split nodes with more than `elements_per_node` elements
    ///
    /// Call after large numbers of `insert`s to maintain consistent search performance.
    pub fn balance(
        &mut self,
        elements_per_node: usize,
        mut get_bounds: impl FnMut(&T) -> Bounds<DIM>,
    ) {
        fn split<const DIM: usize, T>(
            scale: f64,
            embedding: &Embedding<DIM>,
            elements: &mut Slab<Element<T>>,
            level: u32,
            node: &mut Node<DIM>,
            mut get_bounds: impl FnMut(&T) -> Bounds<DIM>,
        ) {
            if level == 0 {
                return;
            }
            let children = ensure_children(&mut node.children);
            let mut next_elt = node.first_element;
            let mut prev_elt = None;
            while let Some(element) = next_elt {
                next_elt = elements[element].next;
                let bounds =
                    embedding.bounds_from_world(scale, &get_bounds(&elements[element].value));
                let Some(index) = bounds.index_in(level - 1) else {
                    // Too large to move into children
                    prev_elt = Some(element);
                    continue;
                };
                // Link into child
                link(elements, &mut children[index], element);
                // Unlink from `node`
                let prev_link = match prev_elt {
                    None => &mut node.first_element,
                    Some(x) => &mut elements[x].next,
                };
                *prev_link = next_elt;
                node.elements -= 1;
            }
        }

        if let Some(root) = &mut self.root {
            // See comment on `DepthFirstTraversal::queue`
            let mut stack = ArrayVec::<(u32, &mut [Node<DIM>]), MAX_DEPTH>::new();
            if root.node.elements > elements_per_node {
                split::<DIM, T>(
                    self.scale,
                    &root.embedding,
                    &mut self.elements,
                    root.coords.level,
                    &mut root.node,
                    &mut get_bounds,
                );
            }
            if let Some(children) = root.node.children.as_mut() {
                stack.push((root.coords.level, children));
            }
            while let Some((level, children)) = stack.pop() {
                let level = level - 1;
                for child in children.iter_mut() {
                    // Balance elements in `child`
                    if child.elements > elements_per_node {
                        split::<DIM, T>(
                            self.scale,
                            &root.embedding,
                            &mut self.elements,
                            level,
                            child,
                            &mut get_bounds,
                        );
                    }

                    // Queue grandchildren for balancing
                    if let Some(grandchildren) = child.children.as_mut() {
                        stack.push((level, grandchildren));
                    }
                }
            }
        }
    }

    /// Remove the value associated with `id`
    pub fn remove(&mut self, bounds: Bounds<DIM>, id: usize) -> T {
        let elt = self.elements.remove(id);
        let root = self.root.as_mut().unwrap();
        let target = root
            .embedding
            .bounds_from_world(self.scale, &bounds)
            .location();
        // A value is guaranteed to be stored in the smallest existing node permitted for it, because:
        // - `insert` only introduces nodes that are siblings of or larger than the root
        // - `balance` always moves all possible elements into newly created child nodes
        let node = find_smallest_existing_parent::<DIM>(root, target);
        unlink(&mut self.elements, node, id);
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
    pub fn intersections(&self, bounds: Bounds<DIM>) -> Intersections<'_, DIM, T> {
        let mut out = Intersections {
            elements: &self.elements,
            traversal: DepthFirstTraversal::default(),
            next_element: None,
        };
        if let Some(ref root) = self.root {
            let bounds = root.embedding.bounds_from_world(self.scale, &bounds);
            if bounds.intersects(&root.coords.bounds()) {
                out.next_element = root.node.first_element;
                if let Some(children) = root.node.children.as_ref() {
                    out.traversal = DepthFirstTraversal {
                        queue: [IntersectingChildren::new(&bounds, root.coords, children)]
                            .into_iter()
                            .collect(),
                        context: bounds,
                    };
                }
            }
        }
        out
    }
}

impl<const DIM: usize, T> Default for SieveTree<DIM, T> {
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
struct Root<const DIM: usize> {
    embedding: Embedding<DIM>,
    coords: CellCoords<DIM>,
    node: Node<DIM>,
}

impl<const DIM: usize> Root<DIM> {
    fn world_bounds(&self, scale: f64) -> Bounds<DIM> {
        self.embedding
            .world_bounds_from_tree(scale, &self.coords.bounds())
    }
}

/// Look up the smallest existing parent of `target`, uprooting the tree if necessary
fn find_smallest_parent<const DIM: usize>(
    root: &mut Root<DIM>,
    target: CellCoords<DIM>,
) -> &mut Node<DIM> {
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
        let children = ensure_children(&mut current.children);
        current_level -= 1;
        let index = child_index_at_level::<DIM>(old_root_coords.min, current_level);
        current = &mut children[index];
    }
    *current = old_root;

    if target.level < ancestor.level {
        // Return the child of the new root that encloses `target`
        let index = child_index_at_level::<DIM>(target.min, root.coords.level - 1);
        &mut root.node.children.as_mut().unwrap()[index]
    } else {
        // Ancestor is target
        &mut root.node
    }
}

fn find_smallest_existing_parent<'a, const DIM: usize>(
    root: &'a mut Root<DIM>,
    target: CellCoords<DIM>,
) -> &'a mut Node<DIM> {
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
                current = mem::transmute::<&mut Node<DIM>, &'a mut Node<DIM>>(&mut children[index]);
            }
        }
    }
    current
}

/// Add `element` to `node`
fn link<const DIM: usize, T>(
    elements: &mut Slab<Element<T>>,
    node: &mut Node<DIM>,
    element: usize,
) {
    let prev = mem::replace(&mut node.first_element, Some(element));
    elements[element].next = prev;
    node.elements += 1;
}

/// Remove `element` from `node`
fn unlink<const DIM: usize, T>(
    elements: &mut Slab<Element<T>>,
    node: &mut Node<DIM>,
    element: usize,
) {
    let successor = elements[element].next;
    let mut link = &mut node.first_element;
    loop {
        let i = link.expect("element missing from node list");
        if i == element {
            *link = successor;
            break;
        }
        link = &mut elements[i].next;
    }
    node.elements -= 1;
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
        local_coords
            .into_iter()
            .enumerate()
            .map(|(i, x)| x as usize * (SUBDIV as usize).pow(i as u32))
            .sum()
    }

    fn children_overlapping(&self, bounds: &TreeBounds<DIM>) -> Option<NodesWithin<DIM>> {
        let mut range = self.bounds().intersection(bounds);
        let level = self.level.checked_sub(1)?;
        let extent = cell_extent(level);
        // Clamp lower bound to `level` grid, then subtract one step in each dimension to allow for
        // edge-crossing
        range.min = range.min.map(|x| (x - x % extent).saturating_sub(extent));
        Some(NodesWithin {
            range,
            cursor: range.min,
            level,
        })
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
struct NodesWithin<const DIM: usize> {
    range: TreeBounds<DIM>,
    cursor: [u64; DIM],
    level: u32,
}

impl<const DIM: usize> Iterator for NodesWithin<DIM> {
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

impl<const DIM: usize> Default for NodesWithin<DIM> {
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
    local_coords
        .into_iter()
        .enumerate()
        .map(|(i, x)| x as usize * (SUBDIV as usize).pow(i as u32))
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

    /// Find the smallest node that a value with these bounds could be stored in, i.e. the largest
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
    /// or `None` if the value must be stored at a higher level
    fn index_in(&self, level: u32) -> Option<usize> {
        let Some(max_extent) = self.extents().into_iter().max() else {
            // 0-dimensional case
            return Some(0);
        };
        let level_extent = cell_extent(level);
        if max_extent > level_extent * u64::from(SUBDIV) {
            return None;
        }
        let local_coords = self.min.map(|x| (x / level_extent) % SUBDIV as u64);
        Some(
            local_coords
                .into_iter()
                .enumerate()
                .map(|(i, x)| x as usize * (SUBDIV as usize).pow(i as u32))
                .sum(),
        )
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
}

impl<const DIM: usize> Default for TreeBounds<DIM> {
    fn default() -> Self {
        Self {
            min: [0; DIM],
            max: [0; DIM],
        }
    }
}

struct DepthFirstTraversal<ChildIter, Context> {
    // By tracking groups of children, rather than individual nodes, we can keep the stack size
    // to O(depth), whereas a naive depth-first traversal would require O(depth * branch^dim).
    queue: ArrayVec<ChildIter, MAX_DEPTH>,
    context: Context,
}

impl<'a, const DIM: usize, I> Iterator for DepthFirstTraversal<I, I::Context>
where
    // `+ Iterator<...>` is redundant here, but rustc insists...
    I: NodeIter<'a, DIM> + Iterator<Item = (CellCoords<DIM>, &'a Node<DIM>)>,
{
    type Item = (CellCoords<DIM>, &'a Node<DIM>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.queue.last_mut()?;
            let Some((coords, node)) = next.next() else {
                self.queue.pop();
                continue;
            };
            if let Some(children) = node.children.as_ref() {
                self.queue
                    .push(NodeIter::new(&self.context, coords, children));
            }
            return Some((coords, node));
        }
    }
}

impl<I, C> Default for DepthFirstTraversal<I, C>
where
    C: Default,
{
    fn default() -> Self {
        Self {
            queue: ArrayVec::new(),
            context: C::default(),
        }
    }
}

trait NodeIter<'a, const DIM: usize>
where
    Self: Iterator<Item = (CellCoords<DIM>, &'a Node<DIM>)>,
{
    type Context;
    fn new(context: &Self::Context, coords: CellCoords<DIM>, children: &'a [Node<DIM>]) -> Self;
}

struct IntersectingChildren<'a, const DIM: usize> {
    children: &'a [Node<DIM>],
    inner: NodesWithin<DIM>,
}

impl<'a, const DIM: usize> Iterator for IntersectingChildren<'a, DIM> {
    type Item = (CellCoords<DIM>, &'a Node<DIM>);

    fn next(&mut self) -> Option<Self::Item> {
        let child_coords = self.inner.next()?;
        let child = &self.children[child_coords.index_in_parent()];
        Some((child_coords, child))
    }
}

impl<'a, const DIM: usize> NodeIter<'a, DIM> for IntersectingChildren<'a, DIM> {
    type Context = TreeBounds<DIM>;

    fn new(bounds: &TreeBounds<DIM>, coords: CellCoords<DIM>, children: &'a [Node<DIM>]) -> Self {
        Self {
            children,
            inner: coords.children_overlapping(bounds).unwrap(),
        }
    }
}

/// Iterator over nodes that might intersect with a [`Bounds`]
pub struct Intersections<'a, const DIM: usize, T> {
    elements: &'a Slab<Element<T>>,
    traversal: DepthFirstTraversal<IntersectingChildren<'a, DIM>, TreeBounds<DIM>>,
    next_element: Option<usize>,
}

impl<'a, const DIM: usize, T> Iterator for Intersections<'a, DIM, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Traverse the current node's elements
            if let Some(index) = self.next_element.take() {
                let elt = &self.elements[index];
                self.next_element = elt.next;
                return Some((index, &elt.value));
            }
            // If the current node has no elements, find a new node
            let (_, node) = self.traversal.next()?;
            self.next_element = node.first_element;
        }
    }
}

#[derive(Debug, Default)]
struct Node<const DIM: usize> {
    // This should become `Box<[Node; SUBDIV.pow(DIM)]>` as soon as Rust permits that
    children: Option<Box<[Node<DIM>]>>,
    /// Length of elements associated directly with this node
    // TODO: Count only unsieved elements
    elements: usize,
    first_element: Option<usize>,
}

fn ensure_children<const DIM: usize>(children: &mut Option<Box<[Node<DIM>]>>) -> &mut [Node<DIM>] {
    children.get_or_insert_with(|| {
        (0..SUBDIV.pow(DIM as u32) as usize)
            .map(|_| Node::default())
            .collect()
    })
}

#[derive(Debug)]
struct Element<T> {
    value: T,
    next: Option<usize>,
}

/// A tree of branching factor 2 covering the entire range of u64 can be at most this deep before
/// nodes can no longer be subdivided.
const MAX_DEPTH: usize = u64::MAX.ilog(2) as usize;

/// Each level subdivides space into this many parts along each dimension
const SUBDIV: u32 = 2;

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
        let mut t = SieveTree::<2, Bounds<2>>::new();
        for y in -5..5 {
            for x in -5..5 {
                let b = Bounds::point([x as f64, y as f64]);
                t.insert(b, b);
            }
        }
        t.balance(1, |&x| x);
        let mut nonempty_nodes = 0;
        for (coords, node) in nodes(&t) {
            assert!(node.elements <= 1, "too many elements at {}", coords);
            if node.elements == 1 {
                assert!(node.children.is_none());
                nonempty_nodes += 1;
            }
        }
        assert_eq!(nonempty_nodes, 100);
        assert!(nodes(&t).count() > 100);
    }

    #[test]
    fn smoke() {
        let mut t = SieveTree::<2, Bounds<2>>::new();
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

    fn nodes<const DIM: usize, T>(
        tree: &SieveTree<DIM, T>,
    ) -> impl Iterator<Item = (u32, &'_ Node<DIM>)> {
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
}
