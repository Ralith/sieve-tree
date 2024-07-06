use core::{array, f64, iter, slice};

use arrayvec::ArrayVec;
use slab::Slab;

use crate::{
    cell_coords::cell_extent, grid_ray::GridRay, grid_size, index_from_local_coords, Cell,
    CellCoords, CellsWithin, Element, Node, Root, TreeBounds, MAX_DEPTH, SUBDIV,
};

#[derive(Default)]
pub struct DepthFirstTraversal<'a, const DIM: usize, const GRID_EXPONENT: u32> {
    // By tracking groups of children, rather than individual nodes, we can keep the stack size
    // to O(depth), whereas a naive depth-first traversal would require O(depth * branch^dim).
    queue: ArrayVec<(CellCoords<DIM>, &'a [Node<DIM, GRID_EXPONENT>]), MAX_DEPTH>,
}

impl<'a, const DIM: usize, const GRID_EXPONENT: u32> DepthFirstTraversal<'a, DIM, GRID_EXPONENT> {
    pub fn push(&mut self, coords: CellCoords<DIM>, node: &'a Node<DIM, GRID_EXPONENT>) {
        if let Some(children) = node.state.children() {
            self.queue.push((coords, children));
        }
    }

    pub fn pop(&mut self) -> Option<(CellCoords<DIM>, &'a [Node<DIM, GRID_EXPONENT>])> {
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

impl<'a, const DIM: usize, const GRID_EXPONENT: u32, T> Intersections<'a, DIM, GRID_EXPONENT, T> {
    pub(crate) fn new(
        maybe_bounds: Option<TreeBounds<DIM>>,
        elements: &'a Slab<Element<T>>,
        root: Option<&'a Root<DIM, GRID_EXPONENT>>,
    ) -> Self {
        let bounds = maybe_bounds.unwrap_or(TreeBounds {
            min: [0; DIM],
            max: [0; DIM],
        });

        let mut out = Intersections {
            bounds,
            elements,
            traversal: DepthFirstTraversal::default(),
            next_child: CellsWithin::default(),
            children: &[],
            next_cell: CellsWithin::default(),
            next_element: ElementIter::default(),
            grid: &[],
        };

        // Skip traversal if the provided bounds are empty
        if maybe_bounds.is_none() {
            return out;
        }

        if let Some(root) = root {
            if bounds.intersects(&root.coords.bounds()) {
                out.traversal.push(root.coords, &root.node);
                out.grid = &root.node.grid;
                out.next_cell = root.coords.cells_overlapping::<GRID_EXPONENT>(&bounds);
            }
        }
        out
    }
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
                    self.grid[coords.index_in_grid::<GRID_EXPONENT>()]
                        .first_element
                        .get(),
                );
                continue;
            }
            // If we're out of cells, get the next child node
            if let Some(coords) = self.next_child.next() {
                let index = coords.index_in_parent();
                let child = &self.children[index];
                if child.elements == 0 {
                    continue;
                }
                self.traversal.push(coords, child);
                self.grid = &child.grid;
                self.next_cell = coords.cells_overlapping::<GRID_EXPONENT>(&self.bounds);
                continue;
            }
            // If we're out of child nodes, get the next group from the queue
            let (coords, children) = self.traversal.pop()?;
            self.children = children;
            self.next_child = coords.children_overlapping(&self.bounds).unwrap();
        }
    }
}

#[derive(Default)]
pub struct ElementIter {
    next: Option<usize>,
}

impl ElementIter {
    pub fn new(next: Option<usize>) -> Self {
        Self { next }
    }

    pub fn next<T>(&mut self, elements: &Slab<Element<T>>) -> Option<usize> {
        let i = self.next?;
        self.next = elements[i].next.get();
        Some(i)
    }
}

/// Iterator over nodes that might overlap with a ray
pub struct Ray<'a, const DIM: usize, const GRID_EXPONENT: u32, T> {
    origin: [f64; DIM],
    direction: [f64; DIM],

    elements: &'a Slab<Element<T>>,
    traversal: DepthFirstTraversal<'a, DIM, GRID_EXPONENT>,
    parent: CellCoords<DIM>,
    children: iter::Enumerate<slice::Iter<'a, Node<DIM, GRID_EXPONENT>>>,
    grid: &'a [Cell],
    grid_coords: CellCoords<DIM>,
    grid_ray: GridRay<DIM>,
    grid_t_max: f64,
    next_element: ElementIter,
}

impl<'a, const DIM: usize, const GRID_EXPONENT: u32, T> Ray<'a, DIM, GRID_EXPONENT, T> {
    pub(crate) fn new(
        elements: &'a Slab<Element<T>>,
        root: Option<&'a Root<DIM, GRID_EXPONENT>>,
        origin: [f64; DIM],
        direction: [f64; DIM],
    ) -> Self {
        let mut out = Self {
            origin: [0.0; DIM],
            direction,

            elements,
            traversal: DepthFirstTraversal::default(),
            parent: CellCoords::from_point([0; DIM], 0),
            children: [].iter().enumerate(),
            grid: &[],
            grid_coords: CellCoords::from_point([0; DIM], 0),
            grid_ray: GridRay::new([0.0; DIM], [0.0; DIM]),
            grid_t_max: f64::NEG_INFINITY,
            next_element: ElementIter::default(),
        };
        if let Some(root) = root {
            // TODO: Fast-path ray vs. root bounds
            out.origin = array::from_fn(|axis| origin[axis] - root.embedding.origin[axis]);
            out.set_grid(root.coords, &root.node);
        }
        out
    }

    /// Configure the iterator to traverse the grid associated with `node`, located at `coords`
    ///
    /// Returns whether the grid can be traversed.
    fn set_grid(&mut self, coords: CellCoords<DIM>, node: &'a Node<DIM, GRID_EXPONENT>) -> bool {
        if node.elements == 0 {
            return false;
        }
        let Some((ray, t_max)) =
            ray_cell::<DIM, GRID_EXPONENT>(&coords, &self.origin, &self.direction)
        else {
            return false;
        };
        self.traversal.push(coords, node);
        self.grid = &node.grid;
        self.grid_coords = coords;
        self.grid_ray = ray;
        self.grid_t_max = t_max;
        true
    }
}

impl<'a, const DIM: usize, const GRID_EXPONENT: u32, T> Iterator
    for Ray<'a, DIM, GRID_EXPONENT, T>
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
            if self.grid_ray.peek_t() < self.grid_t_max {
                let coords = self.grid_ray.next();
                let i = index_from_local_coords(&coords, grid_size::<GRID_EXPONENT>() as u64);
                self.next_element = ElementIter::new(self.grid[i].first_element.get());
                continue;
            }
            // If we've exhausted the current grid, find the next
            if let Some((i, child)) = self.children.next() {
                let coords = self.parent.child(i).unwrap();
                if self.set_grid(coords, child) {
                    continue;
                }
            }

            // If we've exhausted the current node, try the next one
            let (parent, children) = self.traversal.pop()?;
            self.parent = parent;
            self.children = children.iter().enumerate();
        }
    }
}

/// Construct a [GridRay] traversing `cell`, iff they overlap
fn ray_cell<const DIM: usize, const GRID_EXPONENT: u32>(
    cell: &CellCoords<DIM>,
    origin: &[f64; DIM],
    direction: &[f64; DIM],
) -> Option<(GridRay<DIM>, f64)> {
    let bounds = cell.bounds();
    let (min, max) = (0..DIM).fold((f64::NEG_INFINITY, f64::INFINITY), |(min, max), axis| {
        let t1 = (bounds.min[axis] as f64 - origin[axis]) / direction[axis];
        let t2 = ((bounds.max[axis] + 1) as f64 - origin[axis]) / direction[axis];
        (min.max(t1.min(t2)), max.min(t1.max(t2)))
    });
    if max < min || max < 0.0 {
        return None;
    }
    // Compute the local origin in this node, in grid units
    let factor = 1.0 / cell_extent(cell.level - GRID_EXPONENT) as f64;
    let t = min.max(0.0);
    let highest_in_bounds_coordinate = next_down(SUBDIV.pow(GRID_EXPONENT) as f64);
    let origin = array::from_fn(|axis| {
        (((origin[axis] + direction[axis] * t) - bounds.min[axis] as f64) * factor)
            .min(highest_in_bounds_coordinate)
    });
    Some((GridRay::new(origin, *direction), (max - t) * factor))
}

/// Largest f64 smaller than `x`, assuming `x` is finite
const fn next_down(x: f64) -> f64 {
    const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
    const TINY_BITS: u64 = 0x1;
    const NEG_TINY_BITS: u64 = TINY_BITS | SIGN_MASK;

    let bits = x.to_bits();
    let abs = bits & !SIGN_MASK;
    let next_bits = if abs == 0 {
        NEG_TINY_BITS
    } else if bits == abs {
        bits - 1
    } else {
        bits + 1
    };
    f64::from_bits(next_bits)
}
