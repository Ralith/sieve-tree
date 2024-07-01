use arrayvec::ArrayVec;
use slab::Slab;

use crate::{Cell, CellCoords, CellsWithin, Element, Node, Root, TreeBounds, MAX_DEPTH};

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
        bounds: TreeBounds<DIM>,
        elements: &'a Slab<Element<T>>,
        root: Option<&'a Root<DIM, GRID_EXPONENT>>,
    ) -> Self {
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
