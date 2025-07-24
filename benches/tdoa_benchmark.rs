use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use heapless::FnvIndexMap;
use locate_rs::*;
use nalgebra::Vector3;
use rand::distr::{Distribution, Uniform};
use rand::prelude::*;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

fn tdoa_benchmark(c: &mut Criterion) {
    // Generate random position
    let between: Uniform<f64> = Uniform::new_inclusive(-100.0, 100.0).unwrap();
    let r = Arc::new(Mutex::new(StdRng::seed_from_u64(0)));

    c.bench_function("tdoa_3d_4anc_multiple_random_benchmark", |b| {
        b.iter_batched(
            {
                let r = r.clone();
                move || {
                    let mut r = r.lock().unwrap();

                    // 1. Define anchor positions
                    let p1 = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let p2 = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let p3 = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let p4 = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );

                    let mut known_locations: FnvIndexMap<u8, Vector3<f64>, 4> = FnvIndexMap::new();
                    known_locations.insert(1, p1).unwrap();
                    known_locations.insert(2, p2).unwrap();
                    known_locations.insert(3, p3).unwrap();
                    known_locations.insert(4, p4).unwrap();

                    // 2. Ground truth position
                    let x_gt = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );

                    // 3. Compute true distances and TDoAs relative to anchor "1"
                    let r1 = (x_gt - p1).norm();
                    let r2 = (x_gt - p2).norm();
                    let r3 = (x_gt - p3).norm();
                    let r4 = (x_gt - p4).norm();

                    let r21 = r2 - r1;
                    let r31 = r3 - r1;
                    let r41 = r4 - r1;

                    let mut tdoa_infos = FnvIndexMap::<(u8, u8), f64, 4>::new();
                    tdoa_infos.insert((2, 1), r21).unwrap();
                    tdoa_infos.insert((3, 1), r31).unwrap();
                    tdoa_infos.insert((4, 1), r41).unwrap();

                    (known_locations, tdoa_infos)
                }
            },
            {
                |(known_locations, tdoa_infos)| {
                    let mut solver = LocationSolver::new(&known_locations, 1e-6);
                    solver.tdoa(black_box(tdoa_infos), None)
                }
            },
            BatchSize::PerIteration,
        )
    });
}

criterion_group!(benches, tdoa_benchmark);
criterion_main!(benches);
