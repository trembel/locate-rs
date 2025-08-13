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
fn tdoa_3d_4anc_single_f32() {
    // 1. Define anchor positions
    let p1 = Vector3::new(0.0, 0.0, 0.0);
    let p2 = Vector3::new(10.0, 0.0, 0.0);
    let p3 = Vector3::new(0.0, 10.0, 10.0);
    let p4 = Vector3::new(0.0, 0.0, 10.0);

    let mut known_locations = FnvIndexMap::<u8, Vector3<f32>, 4>::new();
    known_locations.insert(1, p1).unwrap();
    known_locations.insert(2, p2).unwrap();
    known_locations.insert(3, p3).unwrap();
    known_locations.insert(4, p4).unwrap();

    // 2. Ground truth position
    let x_gt = Vector3::new(3.4, 4.5, 6.7);

    // 3. Compute true distances and TDoAs relative to anchor "1"
    let r1 = (x_gt - p1).norm();
    let r2 = (x_gt - p2).norm();
    let r3 = (x_gt - p3).norm();
    let r4 = (x_gt - p4).norm();

    let r21 = r2 - r1;
    let r31 = r3 - r1;
    let r41 = r4 - r1;

    let mut tdoa_infos = FnvIndexMap::<(u8, u8), f32, 4>::new();
    tdoa_infos.insert((2, 1), r21).unwrap();
    tdoa_infos.insert((3, 1), r31).unwrap();
    tdoa_infos.insert((4, 1), r41).unwrap();

    // 4. Create solver and solve
    let solving_tolerance = 1e-6;
    let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
    let result = solver.tdoa(tdoa_infos, None);

    // 5. Print and assert results
    assert!(result.is_ok());
    let x_est = result.unwrap();

    // Check that the error is small
    assert!((x_est - x_gt).norm() < solving_tolerance);
}

#[test]
fn tdoa_3d_4anc_single_f64() {
    // 1. Define anchor positions
    let p1 = Vector3::new(0.0, 0.0, 0.0);
    let p2 = Vector3::new(10.0, 0.0, 0.0);
    let p3 = Vector3::new(0.0, 10.0, 10.0);
    let p4 = Vector3::new(0.0, 0.0, 10.0);

    let mut known_locations = FnvIndexMap::<u8, Vector3<f64>, 4>::new();
    known_locations.insert(1, p1).unwrap();
    known_locations.insert(2, p2).unwrap();
    known_locations.insert(3, p3).unwrap();
    known_locations.insert(4, p4).unwrap();

    // 2. Ground truth position
    let x_gt = Vector3::new(3.4, 4.5, 6.7);

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

    // 4. Create solver and solve
    let solving_tolerance = 1e-6;
    let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
    let result = solver.tdoa(tdoa_infos, None);

    // 5. Print and assert results
    assert!(result.is_ok());
    let x_est = result.unwrap();

    // Check that the error is small
    assert!((x_est - x_gt).norm() < solving_tolerance);
}

#[test]
fn tdoa_3d_4anc_multiple_f32() {
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

        let mut known_locations = FnvIndexMap::<u8, Vector3<f32>, 4>::new();
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

        // 3. Compute true distances and TDoAs relative to anchor "1"
        let r1 = (x_gt - p1).norm();
        let r2 = (x_gt - p2).norm();
        let r3 = (x_gt - p3).norm();
        let r4 = (x_gt - p4).norm();

        let r21 = r2 - r1;
        let r31 = r3 - r1;
        let r41 = r4 - r1;

        let mut tdoa_infos = FnvIndexMap::<(u8, u8), f32, 4>::new();
        tdoa_infos.insert((2, 1), r21).unwrap();
        tdoa_infos.insert((3, 1), r31).unwrap();
        tdoa_infos.insert((4, 1), r41).unwrap();

        // 4. Create solver and solve
        let solving_tolerance = 1e-6;
        let mut solver = LocationSolver::new(&known_locations, solving_tolerance);

        match solver.tdoa(tdoa_infos, None) {
            Ok(result) => {
                if (x_gt - result).norm() > 0.001 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert!(innacc_count < 100); // account for very bad anchor placements
    assert_eq!(fail_count, 0);
}

#[test]
fn tdoa_3d_4anc_multiple_f64() {
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

        let mut known_locations = FnvIndexMap::<u8, Vector3<f64>, 4>::new();
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

        // 4. Create solver and solve
        let solving_tolerance = 1e-6;
        let mut solver = LocationSolver::new(&known_locations, solving_tolerance);

        match solver.tdoa(tdoa_infos, None) {
            Ok(result) => {
                if (x_gt - result).norm() > 0.001 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert!(innacc_count < 100); // account for very bad anchor placements
    assert_eq!(fail_count, 0);
}
