#[cfg(test)]
use core::fmt;
use core::mem;

use crate::{grid_size, index_from_local_coords, TreeBounds, SUBDIV};

/// Identifies a single cell, anywhere in tree space
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CellCoords<const DIM: usize> {
    /// Point in this cell with the smallest coordinates on each dimension
    pub min: [u64; DIM],
    /// Exponent of `SUBDIV` which is the cell's extent in each dimension
    pub level: u32,
}

impl<const DIM: usize> CellCoords<DIM> {
    pub fn from_point(point: [u64; DIM], level: u32) -> Self {
        let extent = cell_extent(level);
        Self {
            min: point.map(|x| (x / extent) * extent),
            level,
        }
    }

    pub fn bounds(&self) -> TreeBounds<DIM> {
        let extent = cell_extent(self.level);
        TreeBounds {
            min: self.min,
            max: self.min.map(|x| x + extent - 1),
        }
    }

    pub fn parent(&self) -> Self {
        Self::from_point(self.min, self.level + 1)
    }

    pub fn index_in_parent(&self) -> usize {
        let extent = cell_extent(self.level);
        let local_coords = self.min.map(|x| (x / extent) % SUBDIV as u64);
        index_from_local_coords(&local_coords, SUBDIV.into())
    }

    pub fn index_in_grid<const GRID_EXPONENT: u32>(&self) -> usize {
        let extent = cell_extent(self.level);
        let local_coords = self
            .min
            .map(|x| (x / extent) % grid_size::<GRID_EXPONENT>() as u64);
        index_from_local_coords(&local_coords, grid_size::<GRID_EXPONENT>() as u64)
    }

    /// Iterator over child nodes in a node that might overlap with `bounds`
    pub fn children_overlapping(&self, bounds: &TreeBounds<DIM>) -> Option<CellsWithin<DIM>> {
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
    pub fn cells_overlapping<const GRID_EXPONENT: u32>(
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

    pub fn smallest_common_ancestor(&self, other: &Self) -> Self {
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

#[cfg(test)]
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
pub struct CellsWithin<const DIM: usize> {
    range: TreeBounds<DIM>,
    cursor: [u64; DIM],
    level: u32,
}

impl<const DIM: usize> CellsWithin<DIM> {
    pub fn new(range: TreeBounds<DIM>, level: u32) -> Self {
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

pub const fn cell_extent(level: u32) -> u64 {
    (SUBDIV as u64).saturating_pow(level)
}
