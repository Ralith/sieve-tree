use std::array;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use rand_distr::Exp;

use sieve_tree::*;

pub fn bench(c: &mut Criterion) {
    let mut rng = rand::rngs::SmallRng::from_seed([0xAB; 32]);
    c.bench_function("insert 1000 points and balance", |b| {
        b.iter(|| {
            let mut t = SieveTree::<2, Bounds<2>>::new();
            for _ in 0..1000 {
                let min = array::from_fn(|_| rng.gen_range(-1_000.0..1_000.0));
                let b = Bounds { min, max: min };
                t.insert(b, b);
            }
            t.balance(16, |x| *x);
        });
    });

    c.bench_function("insert 1000 rects and balance", |b| {
        b.iter(|| {
            let mut t = SieveTree::<2, Bounds<2>>::new();
            let width_distr = Exp::new(0.5).unwrap();
            for _ in 0..1000 {
                let min = array::from_fn(|_| rng.gen_range(-1_000.0..1_000.0));
                let width = rng.sample(width_distr);
                let aspect = rng.gen_range(0.2..5.0);
                let height = width / aspect;
                let b = Bounds {
                    min,
                    max: [min[0] + width, min[1] + height],
                };
                t.insert(b, b);
            }
            t.balance(16, |x| *x);
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
