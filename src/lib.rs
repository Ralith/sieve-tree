#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use core::{array, fmt, mem};

use arrayvec::ArrayVec;
use slab::Slab;

/// A `DIM`-dimensional spatial search tree with a branching factor of `BRANCH.pow(DIM)`
#[derive(Debug)]
pub struct SieveTree<const DIM: usize, const BRANCH: u32, T> {
    root: Option<Root<DIM, BRANCH>>,
    elements: Slab<Element<T>>,
}

impl<const DIM: usize, const BRANCH: u32, T> SieveTree<DIM, BRANCH, T> {
    /// Create an empty tree
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `value` into the best existing location in the tree, returning an ID that can be used
    /// to access it directly
    pub fn insert(&mut self, value: T) -> usize
    where
        T: Bounded<DIM>,
    {
        let bounds = value.bounds();
        let id = self.elements.insert(Element { value, next: None });

        let loc = bounds.location::<BRANCH>();
        let node = match &mut self.root {
            None => {
                &mut self
                    .root
                    .insert(Root {
                        coords: loc,
                        node: Node::default(),
                    })
                    .node
            }
            Some(root) => find_smallest_parent::<DIM, BRANCH>(root, loc),
        };
        link(&mut self.elements, node, id);

        id
    }

    /// Recursively split nodes with more than `elements_per_node` elements
    ///
    /// Call after large numbers of `insert`s to maintain consistent search performance.
    pub fn balance(&mut self, elements_per_node: usize)
    where
        T: Bounded<DIM>,
    {
        fn split<const DIM: usize, const BRANCH: u32, T: Bounded<DIM>>(
            elements: &mut Slab<Element<T>>,
            coords: NodeCoords<DIM, BRANCH>,
            node: &mut Node,
        ) {
            let children = ensure_children::<DIM, BRANCH>(&mut node.children);
            let mut next_elt = node.first_element;
            let mut prev_elt = None;
            while let Some(element) = next_elt {
                next_elt = elements[element].next;
                let bounds = elements[element].value.bounds();
                let Some(index) = bounds.index_in::<BRANCH>(coords.level - 1) else {
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

        // See comment on `DepthFirstTraversal::queue`
        let mut stack = ArrayVec::<(NodeCoords<DIM, BRANCH>, &mut [Node]), MAX_DEPTH>::new();
        if let Some(root) = &mut self.root {
            if root.node.elements > elements_per_node {
                split(&mut self.elements, root.coords, &mut root.node);
            }
            if let Some(children) = root.node.children.as_mut() {
                stack.push((root.coords, children));
            }
        }
        while let Some((coords, children)) = stack.pop() {
            for (i, child) in children.iter_mut().enumerate() {
                let child_coords = coords.child(i).unwrap();
                // Balance elements in `child`
                if child.elements > elements_per_node {
                    split(&mut self.elements, child_coords, child);
                }

                // Queue grandchildren for balancing
                if let Some(grandchildren) = child.children.as_mut() {
                    stack.push((child_coords, grandchildren));
                }
            }
        }
    }

    /// Remove the value associated with `id`
    pub fn remove(&mut self, id: usize) -> T
    where
        T: Bounded<DIM>,
    {
        let elt = self.elements.remove(id);
        let node = find_smallest_parent::<DIM, BRANCH>(
            self.root.as_mut().unwrap(),
            elt.value.bounds().location::<BRANCH>(),
        );
        unlink(&mut self.elements, node, id);
        elt.value
    }

    pub fn bounds(&self) -> Option<Bounds<DIM>> {
        self.root.as_ref().map(|root| root.coords.bounds())
    }

    pub fn get(&self, id: usize) -> Option<&T> {
        Some(&self.elements.get(id)?.value)
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        Some(&mut self.elements.get_mut(id)?.value)
    }

    /// Traverse all elements that might intersect with `bounds`
    pub fn intersections(&self, bounds: Bounds<DIM>) -> Intersections<'_, DIM, BRANCH, T> {
        let mut out = Intersections {
            elements: &self.elements,
            traversal: DepthFirstTraversal::default(),
            next_element: None,
        };
        if let Some(ref root) = self.root {
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

impl<const DIM: usize, const BRANCH: u32, T> Default for SieveTree<DIM, BRANCH, T> {
    fn default() -> Self {
        Self {
            root: None,
            elements: Slab::default(),
        }
    }
}

#[derive(Debug)]
struct Root<const DIM: usize, const BRANCH: u32> {
    coords: NodeCoords<DIM, BRANCH>,
    node: Node,
}

/// Look up the smallest existing parent of `target`, uprooting the tree if necessary
fn find_smallest_parent<'a, const DIM: usize, const BRANCH: u32>(
    root: &'a mut Root<DIM, BRANCH>,
    target: NodeCoords<DIM, BRANCH>,
) -> &'a mut Node {
    let ancestor = root.coords.smallest_common_ancestor(&target);
    if ancestor != root.coords {
        // Create new root that encloses both old root and target
        let old_root = mem::take(&mut root.node);
        let old_root_coords = mem::replace(&mut root.coords, ancestor);

        // Reattach the old root under the new root
        let mut current = &mut root.node;
        let mut current_level = root.coords.level;
        while current_level > old_root_coords.level {
            let children = ensure_children::<DIM, BRANCH>(&mut current.children);
            current_level -= 1;
            let index = child_index_at_level::<DIM, BRANCH>(old_root_coords.min, current_level);
            current = &mut children[index];
        }
        *current = old_root;

        if target.level < ancestor.level {
            // Return the child of the new root that encloses `target`
            let index = child_index_at_level::<DIM, BRANCH>(target.min, root.coords.level - 1);
            return &mut root.node.children.as_mut().unwrap()[index];
        } else {
            // Ancestor is target
            return &mut root.node;
        }
    }

    // Find smallest enclosing node that already exists
    let mut current = &mut root.node;
    let mut current_level = root.coords.level;
    {
        while let Some(ref mut children) = current.children {
            if current_level == target.level {
                break;
            }
            current_level -= 1;
            let index = child_index_at_level::<DIM, BRANCH>(target.min, current_level);
            // Hack around borrowck limitation
            unsafe {
                current = mem::transmute::<&mut Node, &'a mut Node>(&mut children[index]);
            }
        }
    }
    current
}

/// Add `element` to `node`
fn link<T>(elements: &mut Slab<Element<T>>, node: &mut Node, element: usize) {
    let prev = mem::replace(&mut node.first_element, Some(element));
    elements[element].next = prev;
    node.elements += 1;
}

/// Remove `element` from `node`
fn unlink<T>(elements: &mut Slab<Element<T>>, node: &mut Node, element: usize) {
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

/// Identifies a tree node
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct NodeCoords<const DIM: usize, const BRANCH: u32> {
    /// Point in this node with the smallest coordinates on each dimension
    min: [u64; DIM],
    /// Exponent of `BRANCH` which is the node's extent in each dimension
    level: u32,
}

impl<const DIM: usize, const BRANCH: u32> NodeCoords<DIM, BRANCH> {
    fn from_point(point: [u64; DIM], level: u32) -> Self {
        let extent = node_extent::<BRANCH>(level);
        Self {
            min: point.map(|x| (x / extent) * extent),
            level,
        }
    }

    fn bounds(&self) -> Bounds<DIM> {
        let extent = node_extent::<BRANCH>(self.level);
        Bounds {
            min: self.min,
            max: self.min.map(|x| x + extent - 1),
        }
    }

    fn parent(&self) -> Self {
        Self::from_point(self.min, self.level + 1)
    }

    fn index_in_parent(&self) -> usize {
        let extent = node_extent::<BRANCH>(self.level);
        let local_coords = self.min.map(|x| (x / extent) % BRANCH as u64);
        local_coords
            .into_iter()
            .enumerate()
            .map(|(i, x)| x as usize * (BRANCH as usize).pow(i as u32))
            .sum()
    }

    fn child(&self, index: usize) -> Option<Self> {
        let level = self.level.checked_sub(1)?;
        let extent = node_extent::<BRANCH>(level);
        Some(Self {
            level,
            min: array::from_fn(|i| {
                let offset = (index as u64 / (BRANCH as u64).pow(i as u32)) % BRANCH as u64;
                self.min[i] + offset * extent
            }),
        })
    }

    fn children_overlapping(&self, bounds: &Bounds<DIM>) -> Option<NodesWithin<DIM, BRANCH>> {
        let mut range = self.bounds().intersection(bounds);
        let level = self.level.checked_sub(1)?;
        let extent = node_extent::<BRANCH>(level);
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

impl<const DIM: usize, const BRANCH: u32> fmt::Display for NodeCoords<DIM, BRANCH> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extent = node_extent::<BRANCH>(self.level);
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

/// Iterator over all nodes at a certain level with a min point inside a [`Bounds`]
struct NodesWithin<const DIM: usize, const BRANCH: u32> {
    range: Bounds<DIM>,
    cursor: [u64; DIM],
    level: u32,
}

impl<const DIM: usize, const BRANCH: u32> Iterator for NodesWithin<DIM, BRANCH> {
    type Item = NodeCoords<DIM, BRANCH>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.last().unwrap() > self.range.max.last().unwrap() {
            return None;
        }
        let result = NodeCoords {
            level: self.level,
            min: self.cursor,
        };
        let extent = node_extent::<BRANCH>(self.level);
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

impl<const DIM: usize, const BRANCH: u32> Default for NodesWithin<DIM, BRANCH> {
    /// Construct the empty iterator
    fn default() -> Self {
        Self {
            range: Bounds {
                min: [0; DIM],
                max: [0; DIM],
            },
            cursor: [1; DIM],
            level: 0,
        }
    }
}

const fn node_extent<const BRANCH: u32>(level: u32) -> u64 {
    (BRANCH as u64).saturating_pow(level)
}

/// Compute the index of the node at `level` containing `point` in its parent's child array
///
/// Equivalent to `NodeCoords::from_point(point, level).index_in_parent()`
fn child_index_at_level<const DIM: usize, const BRANCH: u32>(
    point: [u64; DIM],
    level: u32,
) -> usize {
    let extent = node_extent::<BRANCH>(level);
    let local_coords = point.map(|x| (x / extent) % BRANCH as u64);
    local_coords
        .into_iter()
        .enumerate()
        .map(|(i, x)| x as usize * (BRANCH as usize).pow(i as u32))
        .sum()
}

// `DIM` should probably be an associated constant, but we can't use those in array lengths yet.
pub trait Bounded<const DIM: usize> {
    fn bounds(&self) -> Bounds<DIM>;
}

/// An axis-aligned bounding box
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Bounds<const DIM: usize> {
    /// Smallest point inside the box
    pub min: [u64; DIM],
    /// Largest point inside the box
    pub max: [u64; DIM],
}

impl<const DIM: usize> Bounds<DIM> {
    /// Construct a rectangle covering a single coordinate
    pub const fn point(p: [u64; DIM]) -> Self {
        Self { min: p, max: p }
    }

    /// Number of points inside the rectangle on each axis
    fn extents(&self) -> [u64; DIM] {
        array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }

    /// Find the smallest node that a value with these bounds could be stored in, i.e. the largest
    /// level with cells smaller than this `Bounds`'s extents on any dimension
    fn location<const BRANCH: u32>(&self) -> NodeCoords<DIM, BRANCH> {
        let Some(extent) = self.extents().into_iter().max() else {
            // 0-dimensional case
            return NodeCoords {
                level: 0,
                min: [0; DIM],
            };
        };

        let level = extent.ilog(BRANCH as u64);
        NodeCoords::from_point(self.min, level)
    }

    /// Compute the index of the node at `level` containing this rect in its parent's child array,
    /// or `None` if the value must be stored at a higher level
    fn index_in<const BRANCH: u32>(&self, level: u32) -> Option<usize> {
        let Some(max_extent) = self.extents().into_iter().max() else {
            // 0-dimensional case
            return Some(0);
        };
        let level_extent = node_extent::<BRANCH>(level);
        if max_extent > level_extent * u64::from(BRANCH) {
            return None;
        }
        let local_coords = self.min.map(|x| (x / level_extent) % BRANCH as u64);
        Some(
            local_coords
                .into_iter()
                .enumerate()
                .map(|(i, x)| x as usize * (BRANCH as usize).pow(i as u32))
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

impl<const DIM: usize> Default for Bounds<DIM> {
    fn default() -> Self {
        Self {
            min: [0; DIM],
            max: [0; DIM],
        }
    }
}

impl<const DIM: usize> Bounded<DIM> for Bounds<DIM> {
    fn bounds(&self) -> Bounds<DIM> {
        *self
    }
}

struct DepthFirstTraversal<ChildIter, Context> {
    // By tracking groups of children, rather than individual nodes, we can keep the stack size
    // to O(depth), whereas a naive depth-first traversal would require O(depth * branch^dim).
    //
    // `MAX_DEPTH` should be computed from `BRANCH` once Rust permits that
    queue: ArrayVec<ChildIter, MAX_DEPTH>,
    context: Context,
}

impl<'a, const DIM: usize, const BRANCH: u32, I> Iterator for DepthFirstTraversal<I, I::Context>
where
    // `+ Iterator<...>` is redundant here, but rustc insists...
    I: NodeIter<'a, DIM, BRANCH> + Iterator<Item = (NodeCoords<DIM, BRANCH>, &'a Node)>,
{
    type Item = (NodeCoords<DIM, BRANCH>, &'a Node);

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

trait NodeIter<'a, const DIM: usize, const BRANCH: u32>
where
    Self: Iterator<Item = (NodeCoords<DIM, BRANCH>, &'a Node)>,
{
    type Context;
    fn new(context: &Self::Context, coords: NodeCoords<DIM, BRANCH>, children: &'a [Node]) -> Self;
}

struct IntersectingChildren<'a, const DIM: usize, const BRANCH: u32> {
    children: &'a [Node],
    inner: NodesWithin<DIM, BRANCH>,
}

impl<'a, const DIM: usize, const BRANCH: u32> Iterator for IntersectingChildren<'a, DIM, BRANCH> {
    type Item = (NodeCoords<DIM, BRANCH>, &'a Node);

    fn next(&mut self) -> Option<Self::Item> {
        let child_coords = self.inner.next()?;
        let child = &self.children[child_coords.index_in_parent()];
        Some((child_coords, child))
    }
}

impl<'a, const DIM: usize, const BRANCH: u32> NodeIter<'a, DIM, BRANCH>
    for IntersectingChildren<'a, DIM, BRANCH>
{
    type Context = Bounds<DIM>;

    fn new(bounds: &Bounds<DIM>, coords: NodeCoords<DIM, BRANCH>, children: &'a [Node]) -> Self {
        Self {
            children,
            inner: coords.children_overlapping(bounds).unwrap(),
        }
    }
}

/// Iterator over nodes that might intersect with a [`Bounds`]
pub struct Intersections<'a, const DIM: usize, const BRANCH: u32, T> {
    elements: &'a Slab<Element<T>>,
    traversal: DepthFirstTraversal<IntersectingChildren<'a, DIM, BRANCH>, Bounds<DIM>>,
    next_element: Option<usize>,
}

impl<'a, const DIM: usize, const BRANCH: u32, T> Iterator for Intersections<'a, DIM, BRANCH, T>
where
    T: Bounded<DIM>,
{
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
struct Node {
    // This should become `Box<[Node; BRANCH.pow(DIM)]>` as soon as Rust permits that
    children: Option<Box<[Node]>>,
    /// Length of elements associated directly with this node
    // TODO: Count only unsieved elements
    elements: usize,
    first_element: Option<usize>,
}

fn ensure_children<const DIM: usize, const BRANCH: u32>(
    children: &mut Option<Box<[Node]>>,
) -> &mut [Node] {
    children.get_or_insert_with(|| {
        (0..BRANCH.pow(DIM as u32) as usize)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_ancestors() {
        type Node1D2 = NodeCoords<1, 2>;

        let min = Node1D2::from_point([0], 0);
        assert_eq!(min.smallest_common_ancestor(&min), min);
        assert_eq!(
            Node1D2::from_point([1], 0).smallest_common_ancestor(&min),
            Node1D2::from_point([0], 1)
        );
        assert_eq!(
            Node1D2::from_point([1], 0).smallest_common_ancestor(&min),
            Node1D2::from_point([0], 1)
        );
        assert_eq!(
            Node1D2::from_point([2], 0).smallest_common_ancestor(&min),
            Node1D2::from_point([0], 2)
        );
        assert_eq!(
            Node1D2::from_point([3], 0).smallest_common_ancestor(&min),
            Node1D2::from_point([0], 2)
        );
    }

    #[test]
    fn index_in_parent_leaf() {
        const BRANCH: u32 = 3;
        let origin = BRANCH * 7;
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                assert_eq!(
                    NodeCoords::<2, BRANCH>::from_point(
                        [(origin + x).into(), (origin + y).into()],
                        0
                    )
                    .index_in_parent(),
                    (y * BRANCH + x) as usize
                );
            }
        }
    }

    #[test]
    fn index_in_parent_mid() {
        assert_eq!(NodeCoords::<1, 2>::from_point([5], 2).index_in_parent(), 1);
    }

    #[test]
    fn balance() {
        let mut t = SieveTree::<2, 4, Bounds<2>>::new();
        for y in 0..10 {
            for x in 0..10 {
                t.insert(Bounds::point([x, y]));
            }
        }
        t.balance(1);
        for (coords, node) in nodes(&t) {
            if coords.level > 0 {
                assert_eq!(node.elements, 0, "unexpected elements at {}", coords);
            } else {
                assert!(node.elements <= 1, "too many elements at {}", coords)
            }
        }
        assert!(nodes(&t).count() > 100);
    }

    #[test]
    fn smoke() {
        let mut t = SieveTree::<2, 4, Bounds<2>>::new();
        t.insert(Bounds {
            min: [4, 4],
            max: [107, 107],
        });
        assert_eq!(
            t.intersections(Bounds {
                min: [0, 0],
                max: [10, 10]
            })
            .count(),
            1
        );
        assert_eq!(
            t.intersections(Bounds {
                min: [10, 20],
                max: [30, 40]
            })
            .count(),
            1
        );

        assert_eq!(
            t.intersections(Bounds {
                min: [1000, 1000],
                max: [1001, 1001],
            })
            .count(),
            0
        );
    }

    fn nodes<const DIM: usize, const BRANCH: u32, T>(
        tree: &SieveTree<DIM, BRANCH, T>,
    ) -> impl Iterator<Item = (NodeCoords<DIM, BRANCH>, &'_ Node)> {
        let mut stack = alloc::vec::Vec::new();
        if let Some(root) = &tree.root {
            stack.push((root.coords, &root.node));
        }
        core::iter::from_fn(move || {
            let (coords, node) = stack.pop()?;
            if let Some(children) = node.children.as_ref() {
                stack.extend(
                    children
                        .iter()
                        .enumerate()
                        .map(|(i, node)| (coords.child(i).unwrap(), node)),
                );
            }
            Some((coords, node))
        })
    }
}
