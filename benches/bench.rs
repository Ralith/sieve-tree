use std::array;

use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use rand::{Rng, SeedableRng};
use rand_distr::Exp;

use sieve_tree::*;

pub fn bench(c: &mut Criterion) {
    const GRID_EXPONENT: u32 = 3;
    let mut rng = rand::rngs::SmallRng::from_seed([0xAB; 32]);
    c.bench_function("insert 1000 points and balance", |b| {
        b.iter(|| {
            let mut t = SieveTree::<2, GRID_EXPONENT, Bounds<2>>::new();
            for _ in 0..1000 {
                let min = array::from_fn(|_| rng.gen_range(-1_000.0..1_000.0));
                let b = Bounds { min, max: min };
                t.insert(b, b);
            }
            t.balance(16, |x| *x);
        });
    });

    let mut group = c.benchmark_group("insert rects and balance");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for count in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter(|| {
                let mut t = SieveTree::<2, GRID_EXPONENT, Bounds<2>>::new();
                let width_distr = Exp::new(0.5).unwrap();
                for _ in 0..count {
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
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
