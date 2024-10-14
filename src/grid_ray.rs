use core::array;

/// Iterator over cells along a ray
///
/// Dimensional generalization of "A Fast Voxel Traversal Algorithm for Ray Tracing" (John
/// Amanatides, Andrew Woo)
pub struct GridRay<const DIM: usize> {
    step: [u64; DIM],
    /// Distance along the ray to reach a cell boundary per axis
    t_max: [f64; DIM],
    /// Distance along ray to traverse exactly one cell per axis
    t_delta: [f64; DIM],
    /// Distance along the ray to the next cell
    t: f64,
    /// Coordinates of the next cell
    cell: [u64; DIM],
}

impl<const DIM: usize> GridRay<DIM> {
    /// Construct a ray starting at `origin` and moving in `direction`
    ///
    /// All coordinates of `origin` must be positive. `direction` need not have unit length. Cells
    /// have edges of length 1.
    pub fn new(origin: [f64; DIM], direction: [f64; DIM]) -> Self {
        let cell = origin.map(|x| x as u64);
        Self {
            step: direction.map(|x| match x >= 0.0 {
                true => 1,
                false => u64::MAX, // -1 under wrapping addition
            }),
            t_delta: direction.map(|x| match x >= 0.0 {
                true => 1.0 / x,
                false => -1.0 / x,
            }),
            // Slab method
            t_max: array::from_fn(|i| {
                let low = (cell[i] as f64 - origin[i]) / direction[i];
                let high = (cell[i] as f64 + 1.0 - origin[i]) / direction[i];
                low.max(high)
            }),
            t: 0.0,
            cell,
        }
    }

    /// Distance along the ray to the cell that will be yielded by the next call to
    /// [`next`](Self::next)
    pub fn peek_t(&self) -> f64 {
        self.t
    }

    /// Yield the coordinates of the next cell along the ray
    pub fn next(&mut self) -> [u64; DIM] {
        let result = self.cell;
        let Some((axis, t)) = self
            .t_max
            .into_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(&b))
        else {
            // 0-dimensional case
            return [0; DIM];
        };
        self.t = t;
        self.t_max[axis] += self.t_delta[axis];
        self.cell[axis] = self.cell[axis].wrapping_add(self.step[axis]);
        result
    }
}

impl<const DIM: usize> Iterator for GridRay<DIM> {
    type Item = [u64; DIM];

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_1d() {
        {
            let mut r = GridRay::new([0.0], [1.0]);
            assert_eq!(r.peek_t(), 0.0);
            assert_eq!(r.next(), [0]);
            assert_eq!(r.peek_t(), 1.0);
            assert_eq!(r.next(), [1]);
            assert_eq!(r.peek_t(), 2.0);
            assert_eq!(r.next(), [2]);
        }
        {
            let mut r = GridRay::new([4.5], [1.0]);
            assert_eq!(r.peek_t(), 0.0);
            assert_eq!(r.next(), [4]);
            assert_eq!(r.peek_t(), 0.5);
            assert_eq!(r.next(), [5]);
            assert_eq!(r.peek_t(), 1.5);
            assert_eq!(r.next(), [6]);
        }
        {
            let mut r = GridRay::new([10.0], [-1.0]);
            assert_eq!(r.peek_t(), 0.0);
            assert_eq!(r.next(), [10]);
            assert_eq!(r.peek_t(), 0.0);
            assert_eq!(r.next(), [9]);
            assert_eq!(r.peek_t(), 1.0);
        }
        {
            let mut r = GridRay::new([4.5], [100.0]);
            assert_eq!(r.peek_t(), 0.0);
            assert_eq!(r.next(), [4]);
            assert_eq!(r.peek_t(), 0.5 / 100.0);
            assert_eq!(r.next(), [5]);
        }
    }

    #[test]
    fn smoke_2d() {
        {
            let mut r = GridRay::new([1.5, 1.5], [1.0, 0.0]);
            assert_eq!(r.next(), [1, 1]);
            assert_eq!(r.peek_t(), 0.5);
            assert_eq!(r.next(), [2, 1]);
        }
        {
            let mut r = GridRay::new([1.5, 1.5], [0.0, 1.0]);
            assert_eq!(r.next(), [1, 1]);
            assert_eq!(r.peek_t(), 0.5);
            assert_eq!(r.next(), [1, 2]);
        }
        {
            let mut r = GridRay::new([1.5, 1.5], [-1.0, 0.0]);
            assert_eq!(r.next(), [1, 1]);
            assert_eq!(r.peek_t(), 0.5);
            assert_eq!(r.next(), [0, 1]);
        }
        {
            let mut r = GridRay::new([1.5, 1.5], [0.0, -1.0]);
            assert_eq!(r.next(), [1, 1]);
            assert_eq!(r.peek_t(), 0.5);
            assert_eq!(r.next(), [1, 0]);
        }
    }

    #[test]
    fn angled() {
        let mut r = GridRay::new([1.5, 1.5], [1.0, 2.0]);
        assert_eq!(r.peek_t(), 0.0);
        assert_eq!(r.next(), [1, 1]);
        assert_eq!(r.peek_t(), 0.25);
        assert_eq!(r.next(), [1, 2]);
        assert_eq!(r.next(), [2, 2]);
        assert_eq!(r.next(), [2, 3]);
    }
}
