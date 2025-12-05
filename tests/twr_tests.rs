use approx::assert_relative_eq;
use heapless::FnvIndexMap;
use locate_rs::LocationSolver;
use nalgebra::{RealField, Vector3};
use num_traits::{Float, float::TotalOrder};
use rand::distr::{Distribution, Uniform};
use rand::prelude::*;

fn bad_placement_tetrahedron_volume<FLOAT>(
    a: &Vector3<FLOAT>,
    b: &Vector3<FLOAT>,
    c: &Vector3<FLOAT>,
    d: &Vector3<FLOAT>,
    tol: FLOAT,
) -> bool
where
    FLOAT: num_traits::float::Float,
    FLOAT: RealField,
    FLOAT: simba::scalar::SubsetOf<f64>,
    FLOAT: TotalOrder,
    FLOAT: std::iter::Sum,
{
    let volume =
        Float::abs(((b - a).cross(&(c - a))).dot(&(d - a))) / FLOAT::from_f64(6.0).unwrap();

    // Compute all 6 edges
    let edges = [
        (b - a).norm(),
        (c - a).norm(),
        (d - a).norm(),
        (c - b).norm(),
        (d - b).norm(),
        (d - c).norm(),
    ];
    let avg_edge = edges.iter().copied().sum::<FLOAT>() / FLOAT::from_usize(edges.len()).unwrap();

    let normalized_volume = if avg_edge < FLOAT::from_f64(1e-6).unwrap() {
        FLOAT::from_f64(0.0).unwrap() // avoid division by 0
    } else {
        volume / Float::powi(avg_edge, 3)
    };

    normalized_volume < tol
}

#[test]
fn twr_3d_4anc_single_f32() {
    let r = StdRng::seed_from_u64(0);
    let x_gt: Vector3<f32> = Vector3::new(3.0, 0.0, 0.0);

    let p1 = Vector3::new(1.0, 0.0, 0.0);
    let p2 = Vector3::new(0.0, 1.0, 0.0);
    let p3 = Vector3::new(0.0, 0.0, 1.0);
    let p4 = Vector3::new(0.5, 0.0, 1.0);

    let mut known_locations: FnvIndexMap<u8, Vector3<f32>, 8> = FnvIndexMap::new();
    known_locations.insert(1, p1).unwrap();
    known_locations.insert(2, p2).unwrap();
    known_locations.insert(3, p3).unwrap();
    known_locations.insert(4, p4).unwrap();

    let mut trilateration_infos: FnvIndexMap<u8, f32, 8> = FnvIndexMap::new();
    for (node, loc) in &known_locations {
        let _ = trilateration_infos.insert(*node, (x_gt - loc).norm());
    }

    let mut solver = LocationSolver::new(&known_locations, 1e-6);
    let result = solver.trilateration(trilateration_infos, r).unwrap();

    assert_relative_eq!(x_gt, result, epsilon = 1e-3); // assert to 1 millimeter accuracy
}

#[test]
fn twr_3d_4anc_single_f32_fast() {
    let x_gt: Vector3<f32> = Vector3::new(3.0, 0.0, 0.0);

    let p1 = Vector3::new(1.0, 0.0, 0.0);
    let p2 = Vector3::new(0.0, 1.0, 0.0);
    let p3 = Vector3::new(0.0, 0.0, 1.0);
    let p4 = Vector3::new(0.5, 0.0, 1.0);

    let mut known_locations: FnvIndexMap<u8, Vector3<f32>, 8> = FnvIndexMap::new();
    known_locations.insert(1, p1).unwrap();
    known_locations.insert(2, p2).unwrap();
    known_locations.insert(3, p3).unwrap();
    known_locations.insert(4, p4).unwrap();

    let mut trilateration_infos: FnvIndexMap<u8, f32, 8> = FnvIndexMap::new();
    for (node, loc) in &known_locations {
        let _ = trilateration_infos.insert(*node, (x_gt - loc).norm());
    }

    let mut solver = LocationSolver::new(&known_locations, 1e-6);
    let result = solver
        .trilateration_fast(trilateration_infos, None)
        .unwrap();

    assert_relative_eq!(x_gt, result, epsilon = 1e-3); // assert to 1 millimeter accuracy
}

#[test]
fn twr_3d_4anc_single_f64() {
    let r = StdRng::seed_from_u64(0);
    let x_gt = Vector3::new(3.0, 0.0, 0.0);

    let p1 = Vector3::new(1.0, 0.0, 0.0);
    let p2 = Vector3::new(0.0, 1.0, 0.0);
    let p3 = Vector3::new(0.0, 0.0, 1.0);
    let p4 = Vector3::new(0.5, 0.0, 1.0);

    let mut known_locations: FnvIndexMap<u8, Vector3<f64>, 8> = FnvIndexMap::new();
    known_locations.insert(1, p1).unwrap();
    known_locations.insert(2, p2).unwrap();
    known_locations.insert(3, p3).unwrap();
    known_locations.insert(4, p4).unwrap();

    let mut trilateration_infos: FnvIndexMap<u8, f64, 8> = FnvIndexMap::new();
    for (node, loc) in &known_locations {
        let _ = trilateration_infos.insert(*node, (x_gt - loc).norm());
    }

    let mut solver = LocationSolver::new(&known_locations, 1e-8);
    let result = solver.trilateration(trilateration_infos, r).unwrap();

    assert_relative_eq!(x_gt, result, epsilon = 1e-3); // assert to 1 millimeter accuracy
}

#[test]
fn twr_3d_4anc_single_f64_fast() {
    let x_gt = Vector3::new(3.0, 0.0, 0.0);

    let p1 = Vector3::new(1.0, 0.0, 0.0);
    let p2 = Vector3::new(0.0, 1.0, 0.0);
    let p3 = Vector3::new(0.0, 0.0, 1.0);
    let p4 = Vector3::new(0.5, 0.0, 1.0);

    let mut known_locations: FnvIndexMap<u8, Vector3<f64>, 8> = FnvIndexMap::new();
    known_locations.insert(1, p1).unwrap();
    known_locations.insert(2, p2).unwrap();
    known_locations.insert(3, p3).unwrap();
    known_locations.insert(4, p4).unwrap();

    let mut trilateration_infos: FnvIndexMap<u8, f64, 8> = FnvIndexMap::new();
    for (node, loc) in &known_locations {
        let _ = trilateration_infos.insert(*node, (x_gt - loc).norm());
    }

    let mut solver = LocationSolver::new(&known_locations, 1e-6);
    let result = solver
        .trilateration_fast(trilateration_infos, None)
        .unwrap();

    assert_relative_eq!(x_gt, result, epsilon = 1e-3); // assert to 1 millimeter accuracy
}

#[test]
fn twr_3d_4anc_multiple_random_f32() {
    let mut r = StdRng::seed_from_u64(0);
    let between = Uniform::new_inclusive(-100.0, 100.0).unwrap();

    let mut fail_count: usize = 0;
    let mut innacc_count: usize = 0;

    for _ in 0..1000 {
        let mut p1: Vector3<f32>;
        let mut p2: Vector3<f32>;
        let mut p3: Vector3<f32>;
        let mut p4: Vector3<f32>;

        loop {
            // 1. Define anchor positions
            p1 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p2 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p3 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p4 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );

            if bad_placement_tetrahedron_volume(&p1, &p2, &p3, &p4, 0.1) == false {
                break;
            }
        }

        let mut known_locations: FnvIndexMap<u8, Vector3<f32>, 4> = FnvIndexMap::new();
        known_locations.insert(1, p1).unwrap();
        known_locations.insert(2, p2).unwrap();
        known_locations.insert(3, p3).unwrap();
        known_locations.insert(4, p4).unwrap();

        // 2. Ground truth position
        let x_gt = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        // 3. Compute true distances for TWR
        let mut trilateration_infos: FnvIndexMap<u8, f32, 4> = FnvIndexMap::new();
        for (node, loc) in &known_locations {
            let _ = trilateration_infos.insert(
                *node,
                (x_gt.cast::<f64>() - loc.cast::<f64>()).norm() as f32,
            );
        }

        // 4. Create solver and solve
        let solving_tolerance = 1e-6;
        let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
        match solver.trilateration(trilateration_infos, r.clone()) {
            Ok(result) => {
                if (x_gt - result).norm() > 1e-3 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert!(innacc_count < 15);
    assert_eq!(fail_count, 0);
}

#[test]
fn twr_3d_4anc_multiple_random_f32_fast() {
    let mut r = StdRng::seed_from_u64(0);
    let between = Uniform::new_inclusive(-100.0, 100.0).unwrap();

    let mut fail_count: usize = 0;
    let mut innacc_count: usize = 0;

    for _ in 0..1000 {
        let mut p1: Vector3<f32>;
        let mut p2: Vector3<f32>;
        let mut p3: Vector3<f32>;
        let mut p4: Vector3<f32>;

        loop {
            // 1. Define anchor positions
            p1 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p2 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p3 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p4 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );

            if bad_placement_tetrahedron_volume(&p1, &p2, &p3, &p4, 0.1) == false {
                break;
            }
        }

        let mut known_locations: FnvIndexMap<u8, Vector3<f32>, 4> = FnvIndexMap::new();
        known_locations.insert(1, p1).unwrap();
        known_locations.insert(2, p2).unwrap();
        known_locations.insert(3, p3).unwrap();
        known_locations.insert(4, p4).unwrap();

        // 2. Ground truth position
        let x_gt = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        // 3. Compute true distances for TWR
        let mut trilateration_infos: FnvIndexMap<u8, f32, 4> = FnvIndexMap::new();
        for (node, loc) in &known_locations {
            let _ = trilateration_infos.insert(
                *node,
                (x_gt.cast::<f64>() - loc.cast::<f64>()).norm() as f32,
            );
        }

        // 4. Create solver and solve
        let solving_tolerance = 1e-6;
        let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
        match solver.trilateration_fast(trilateration_infos, None) {
            Ok(result) => {
                if (x_gt - result).norm() > 1e-3 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert!(innacc_count < 10);
    assert_eq!(fail_count, 0);
}

#[test]
fn twr_3d_4anc_multiple_random_f64() {
    let mut r = StdRng::seed_from_u64(0);
    let between = Uniform::new_inclusive(-100.0, 100.0).unwrap();

    let mut fail_count: usize = 0;
    let mut innacc_count: usize = 0;

    for _ in 0..1000 {
        let mut p1: Vector3<f64>;
        let mut p2: Vector3<f64>;
        let mut p3: Vector3<f64>;
        let mut p4: Vector3<f64>;

        loop {
            // 1. Define anchor positions
            p1 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p2 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p3 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p4 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );

            if bad_placement_tetrahedron_volume(&p1, &p2, &p3, &p4, 0.1) == false {
                break;
            }
        }

        let mut known_locations: FnvIndexMap<u8, Vector3<f64>, 4> = FnvIndexMap::new();
        known_locations.insert(1, p1).unwrap();
        known_locations.insert(2, p2).unwrap();
        known_locations.insert(3, p3).unwrap();
        known_locations.insert(4, p4).unwrap();

        // 2. Ground truth position
        let x_gt = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        // 3. Compute true distances for TWR
        let mut trilateration_infos: FnvIndexMap<u8, f64, 4> = FnvIndexMap::new();
        for (node, loc) in &known_locations {
            let _ =
                trilateration_infos.insert(*node, (x_gt.cast::<f64>() - loc.cast::<f64>()).norm());
        }

        // 4. Create solver and solve
        let solving_tolerance = 1e-6;
        let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
        match solver.trilateration(trilateration_infos, r.clone()) {
            Ok(result) => {
                if (x_gt - result).norm() > 1e-3 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert!(innacc_count < 10);
    assert_eq!(fail_count, 0);
}

#[test]
fn twr_3d_4anc_multiple_random_f64_fast() {
    let mut r = StdRng::seed_from_u64(0);
    let between = Uniform::new_inclusive(-100.0, 100.0).unwrap();

    let mut fail_count: usize = 0;
    let mut innacc_count: usize = 0;

    for _ in 0..1000 {
        let mut p1: Vector3<f64>;
        let mut p2: Vector3<f64>;
        let mut p3: Vector3<f64>;
        let mut p4: Vector3<f64>;

        loop {
            // 1. Define anchor positions
            p1 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p2 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p3 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );
            p4 = Vector3::new(
                between.sample(&mut r),
                between.sample(&mut r),
                between.sample(&mut r),
            );

            if bad_placement_tetrahedron_volume(&p1, &p2, &p3, &p4, 0.1) == false {
                break;
            }
        }

        let mut known_locations: FnvIndexMap<u8, Vector3<f64>, 4> = FnvIndexMap::new();
        known_locations.insert(1, p1).unwrap();
        known_locations.insert(2, p2).unwrap();
        known_locations.insert(3, p3).unwrap();
        known_locations.insert(4, p4).unwrap();

        // 2. Ground truth position
        let x_gt = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        // 3. Compute true distances for TWR
        let mut trilateration_infos: FnvIndexMap<u8, f64, 4> = FnvIndexMap::new();
        for (node, loc) in &known_locations {
            let _ =
                trilateration_infos.insert(*node, (x_gt.cast::<f64>() - loc.cast::<f64>()).norm());
        }

        // 4. Create solver and solve
        let solving_tolerance = 1e-6;
        let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
        match solver.trilateration_fast(trilateration_infos, None) {
            Ok(result) => {
                if (x_gt - result).norm() > 1e-3 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert!(innacc_count < 10);
    assert_eq!(fail_count, 0);
}
