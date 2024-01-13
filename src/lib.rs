#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use core::{array, fmt, mem};

use arrayvec::ArrayVec;
use slab::Slab;

#[derive(Debug)]
pub struct SieveTree<const DIM: usize, const BRANCH: u32, T> {
    root_coords: NodeCoords<DIM, BRANCH>,
    root: Node,
    elements: Slab<Element<T>>,
}

impl<const DIM: usize, const BRANCH: u32, T> SieveTree<DIM, BRANCH, T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, value: T) -> usize
    where
        T: Bounded<DIM>,
    {
        let bounds = value.bounds();
        let id = self.elements.insert(Element { value, next: None });

        let loc = bounds.location::<BRANCH>();
        let node = find_smallest_parent::<DIM, BRANCH>(&mut self.root_coords, &mut self.root, loc);
        link(&mut self.elements, node, id);

        id
    }

    pub fn remove(&mut self, id: usize) -> T
    where
        T: Bounded<DIM>,
    {
        let elt = self.elements.remove(id);
        let node = find_smallest_parent::<DIM, BRANCH>(
            &mut self.root_coords,
            &mut self.root,
            elt.value.bounds().location::<BRANCH>(),
        );
        unlink(&mut self.elements, node, id);
        elt.value
    }

    pub fn get(&self, id: usize) -> Option<&T> {
        Some(&self.elements.get(id)?.value)
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        Some(&mut self.elements.get_mut(id)?.value)
    }

    fn depth_first(&self) -> DepthFirstTraversal<'_, DIM, BRANCH> {
        DepthFirstTraversal {
            queue: [DepthFirstQueueEntry {
                node: self.root_coords,
                children: self.root.children.as_ref().map_or(&[], |x| x),
                index: 0,
            }]
            .into_iter()
            .collect(),
        }
    }

    /// Traverse all elements that intersect with `bounds`
    pub fn intersections<'a>(&'a self, bounds: &'a Rect<DIM>) -> Intersections<'a, DIM, BRANCH, T> {
        let mut out = Intersections {
            bounds,
            elements: &self.elements,
            traversal: DepthFirstTraversal::default(),
            next_element: None,
        };
        if bounds.intersects(&self.root_coords.bounds()) {
            out.next_element = self.root.first_element;
            out.traversal = self.depth_first();
        }
        out
    }
}

impl<const DIM: usize, const BRANCH: u32, T> Default for SieveTree<DIM, BRANCH, T> {
    fn default() -> Self {
        Self {
            root_coords: NodeCoords {
                level: 0,
                min: [0; DIM],
            },
            root: Node::default(),
            elements: Slab::default(),
        }
    }
}

/// Look up the smallest existing parent of `target`, uprooting the tree if necessary
fn find_smallest_parent<'a, const DIM: usize, const BRANCH: u32>(
    root_coords: &mut NodeCoords<DIM, BRANCH>,
    root: &'a mut Node,
    target: NodeCoords<DIM, BRANCH>,
) -> &'a mut Node {
    let ancestor = root_coords.smallest_common_ancestor(&target);
    if ancestor != *root_coords {
        // Create new root, reattach the old root under it, then return the new root
        let old_root = mem::take(root);
        let old_root_coords = mem::replace(root_coords, ancestor);

        let mut current = &mut *root;
        let mut current_level = root_coords.level;
        while current_level > old_root_coords.level {
            let children = current.children.insert(
                (0..BRANCH.pow(DIM as u32) as usize)
                    .map(|_| Node::default())
                    .collect(),
            );
            current_level -= 1;
            let coords = NodeCoords::<DIM, BRANCH>::from_point(old_root_coords.min, current_level);
            current = &mut children[coords.index_in_parent()];
        }
        *current = old_root;

        return root;
    }

    // Find smallest enclosing node that already exists
    let mut current = root;
    let mut current_level = root_coords.level;
    {
        while let Some(ref mut children) = current.children {
            if current_level == target.level {
                break;
            }
            current_level -= 1;
            let child_coords = NodeCoords::<DIM, BRANCH>::from_point(target.min, current_level);
            // Hack around borrowck limitation
            unsafe {
                current = mem::transmute::<&mut Node, &'a mut Node>(
                    &mut children[child_coords.index_in_parent()],
                );
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

    fn bounds(&self) -> Rect<DIM> {
        let extent = node_extent::<BRANCH>(self.level);
        Rect {
            min: self.min,
            max: self.min.map(|x| x + extent - 1),
        }
    }

    fn parent(&self) -> Self {
        Self::from_point(self.min, self.level + 1)
    }

    fn index_in_parent(&self) -> usize {
        let parent_extent = node_extent::<BRANCH>(self.level + 1);
        let local_coords = self.min.map(|x| x % parent_extent);
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

    fn children_overlapping(&self, bounds: &Rect<DIM>) -> Option<NodesWithin<DIM, BRANCH>> {
        let range = self.bounds().intersection(bounds);
        Some(NodesWithin {
            range,
            cursor: range.min,
            level: self.level.checked_sub(1)?,
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
        for level in (0..=self.level).rev() {
            let extent = node_extent::<BRANCH>(level);
            if level != 0 {
                write!(f, ".")?;
            }
            write!(f, "(")?;
            for (i, x) in self.min.into_iter().enumerate() {
                if i != 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", x / extent)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Iterator over all nodes at a certain level with a min point inside a [`Rect`]
struct NodesWithin<const DIM: usize, const BRANCH: u32> {
    range: Rect<DIM>,
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
        self.cursor[0] += 1;
        for i in 1..DIM {
            if self.cursor[i - 1] <= self.range.max[i - 1] {
                break;
            }
            self.cursor[i - 1] = self.range.min[i - 1];
            self.cursor[i] += 1;
        }
        Some(result)
    }
}

const fn node_extent<const BRANCH: u32>(level: u32) -> u64 {
    (BRANCH as u64).saturating_pow(level)
}

pub trait Bounded<const DIM: usize> {
    fn bounds(&self) -> Rect<DIM>;
}

/// An axis-aligned bounding box
// `DIM` should probably be an associated constant, but we can't use those in array lengths yet.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Rect<const DIM: usize> {
    /// Smallest point inside the box
    pub min: [u64; DIM],
    /// Largest point inside the box
    pub max: [u64; DIM],
}

impl<const DIM: usize> Rect<DIM> {
    pub const ROOT: Self = Self {
        min: [0; DIM],
        max: [u64::MAX; DIM],
    };

    /// Number of points inside the rectangle on each axis
    fn extents(&self) -> [u64; DIM] {
        array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }

    /// Find the smallest node that a value with these bounds could be stored in, i.e. the largest
    /// level with cells smaller than this `Rect`'s extents on any dimension
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

impl<const DIM: usize> Default for Rect<DIM> {
    fn default() -> Self {
        Self {
            min: [0; DIM],
            max: [0; DIM],
        }
    }
}

impl<const DIM: usize> Bounded<DIM> for Rect<DIM> {
    fn bounds(&self) -> Rect<DIM> {
        *self
    }
}

#[derive(Default)]
struct DepthFirstTraversal<'a, const DIM: usize, const BRANCH: u32> {
    // `MAX_DEPTH` should be computed from `BRANCH` once Rust permits that
    queue: ArrayVec<DepthFirstQueueEntry<'a, DIM, BRANCH>, MAX_DEPTH>,
}

impl<'a, const DIM: usize, const BRANCH: u32> Iterator for DepthFirstTraversal<'a, DIM, BRANCH> {
    type Item = (NodeCoords<DIM, BRANCH>, &'a Node);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next = self.queue.last_mut()?;
            let Some(node) = next.children.get(next.index) else {
                self.queue.pop();
                continue;
            };
            let coords = next.node;
            let child_coords = coords.child(next.index);
            next.index += 1;
            if let (Some(children), Some(child_coords)) = (node.children.as_ref(), child_coords) {
                self.queue.push(DepthFirstQueueEntry {
                    node: child_coords,
                    children,
                    index: 0,
                });
            }
            return Some((coords, node));
        }
    }
}

struct DepthFirstQueueEntry<'a, const DIM: usize, const BRANCH: u32> {
    node: NodeCoords<DIM, BRANCH>,
    children: &'a [Node],
    index: usize,
}

pub struct Intersections<'a, const DIM: usize, const BRANCH: u32, T> {
    bounds: &'a Rect<DIM>,
    elements: &'a Slab<Element<T>>,
    traversal: DepthFirstTraversal<'a, DIM, BRANCH>,
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
                if !elt.value.bounds().intersects(self.bounds) {
                    continue;
                }
                return Some((index, &elt.value));
            }
            // If the current node has no elements, find a new node
            loop {
                let (coords, node) = self.traversal.next()?;
                if !coords.bounds().intersects(self.bounds) {
                    continue;
                }
                self.next_element = node.first_element;
            }
        }
    }
}

#[derive(Debug, Default)]
struct Node {
    // This should become `Box<[Node; BRANCH.pow(DIM)]>` as soon as Rust permits that
    children: Option<Box<[Node]>>,
    /// Length of elements associated directly with this node
    elements: usize,
    first_element: Option<usize>,
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
    fn index_in_parent() {
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
    fn smoke() {
        let mut t = SieveTree::<2, 4, Rect<2>>::new();
        t.insert(Rect {
            min: [4, 4],
            max: [107, 107],
        });
        assert_eq!(
            t.intersections(&Rect {
                min: [0, 0],
                max: [10, 10]
            })
            .count(),
            1
        );
        assert_eq!(
            t.intersections(&Rect {
                min: [10, 20],
                max: [30, 40]
            })
            .count(),
            1
        );

        assert_eq!(
            t.intersections(&Rect {
                min: [0, 0],
                max: [2, 2]
            })
            .count(),
            0
        );
        assert_eq!(
            t.intersections(&Rect {
                min: [1000, 1000],
                max: [1001, 1001],
            })
            .count(),
            0
        );
    }
}
