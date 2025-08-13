use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use heapless::FnvIndexMap;
use locate_rs::*;
use nalgebra::Vector3;
use rand::distr::{Distribution, Uniform};
use rand::prelude::*;
use std::hint::black_box;
use std::sync::{Arc, Mutex};

fn twr_benchmark(c: &mut Criterion) {
    // Generate random position
    let between: Uniform<f32> = Uniform::new_inclusive(-100.0, 100.0).unwrap();
    let r = Arc::new(Mutex::new(StdRng::seed_from_u64(0)));

    c.bench_function("twr_3d_4anc_multiple_random_benchmark", |b| {
        b.iter_batched(
            {
                let r = r.clone();
                move || {
                    let mut r = r.lock().unwrap();

                    let location_gt = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );

                    let location_a = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let location_b = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let location_c = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let location_d = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );

                    let mut locs: FnvIndexMap<u8, Vector3<f32>, 8> = FnvIndexMap::new();
                    locs.insert(1, location_a).unwrap();
                    locs.insert(2, location_b).unwrap();
                    locs.insert(3, location_c).unwrap();
                    locs.insert(4, location_d).unwrap();

                    let mut trilateration_infos: FnvIndexMap<u8, f32, 8> = FnvIndexMap::new();
                    for (node, loc) in &locs {
                        let _ = trilateration_infos.insert(*node, (location_gt - loc).norm());
                    }

                    (locs, trilateration_infos)
                }
            },
            {
                let r = r.clone();
                move |(locs, trilateration_infos)| {
                    let mut r = r.lock().unwrap();
                    let mut solver = LocationSolver::new(&locs, 1e-6);
                    solver.trilateration(black_box(trilateration_infos), &mut *r)
                }
            },
            BatchSize::PerIteration,
        )
    });
}

fn twr_benchmark_fast(c: &mut Criterion) {
    // Generate random position
    let between: Uniform<f32> = Uniform::new_inclusive(-100.0, 100.0).unwrap();
    let r = Arc::new(Mutex::new(StdRng::seed_from_u64(0)));

    c.bench_function("twr_3d_4anc_multiple_random_benchmark_fast", |b| {
        b.iter_batched(
            {
                let r = r.clone();
                move || {
                    let mut r = r.lock().unwrap();

                    let location_gt = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );

                    let location_a = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let location_b = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let location_c = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );
                    let location_d = Vector3::new(
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                        between.sample(&mut *r),
                    );

                    let mut locs: FnvIndexMap<u8, Vector3<f32>, 8> = FnvIndexMap::new();
                    locs.insert(1, location_a).unwrap();
                    locs.insert(2, location_b).unwrap();
                    locs.insert(3, location_c).unwrap();
                    locs.insert(4, location_d).unwrap();

                    let mut trilateration_infos: FnvIndexMap<u8, f32, 8> = FnvIndexMap::new();
                    for (node, loc) in &locs {
                        let _ = trilateration_infos.insert(*node, (location_gt - loc).norm());
                    }

                    (locs, trilateration_infos)
                }
            },
            {
                |(locs, trilateration_infos)| {
                    let mut solver = LocationSolver::new(&locs, 1e-6);
                    solver.trilateration_fast(black_box(trilateration_infos), None)
                }
            },
            BatchSize::PerIteration,
        )
    });
}

criterion_group!(benches, twr_benchmark, twr_benchmark_fast);
criterion_main!(benches);
