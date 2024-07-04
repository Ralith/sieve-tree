use core::array;
#[cfg(test)]
use core::fmt;

use crate::{cell_extent, index_from_local_coords, level_for_extent, CellCoords, SUBDIV};

/// An axis-aligned bounding box in tree space
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct TreeBounds<const DIM: usize> {
    /// Smallest point inside the box
    pub min: [u64; DIM],
    /// Largest point inside the box
    pub max: [u64; DIM],
}

impl<const DIM: usize> TreeBounds<DIM> {
    /// Number of points inside the rectangle on each axis
    pub fn extents(&self) -> [u64; DIM] {
        array::from_fn(|i| self.max[i] - self.min[i] + 1)
    }

    /// The level component of [`location()`](Self::location)
    pub fn level<const GRID_EXPONENT: u32>(&self) -> u32 {
        let Some(extent) = self.extents().into_iter().max() else {
            // 0-dimensional case
            return 0;
        };

        level_for_extent(extent) + GRID_EXPONENT
    }

    /// Find the smallest node a value with these bounds could be stored in, i.e. the largest level
    /// of nodes with cells smaller than this `Bounds`'s extents on any dimension
    pub fn node_location<const GRID_EXPONENT: u32>(&self) -> CellCoords<DIM> {
        let level = self.level::<GRID_EXPONENT>();
        CellCoords::from_point(self.min, level)
    }

    /// Compute the index of the node at `level` containing this rect in its parent's child array,
    /// and the index in the selected node's grid, or `None` if the value must be stored at a higher
    /// level
    pub fn index_in<const GRID_EXPONENT: u32>(&self, level: u32) -> Option<usize> {
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

    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: array::from_fn(|i| self.min[i].max(other.min[i])),
            max: array::from_fn(|i| self.max[i].min(other.max[i])),
        }
    }

    #[cfg(test)]
    pub fn contains(&self, point: &[u64; DIM]) -> bool {
        self.min
            .iter()
            .zip(&self.max)
            .zip(point)
            .all(|((min, max), x)| min <= x && x <= max)
    }

    /// Shift the minimum value towards the origin by `range`
    pub fn relax(&self, range: u64) -> Self {
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

#[cfg(test)]
impl<const DIM: usize> fmt::Display for TreeBounds<DIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}-{:?}", self.min, self.max)
    }
}
