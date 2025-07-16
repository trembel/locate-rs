use approx::assert_relative_eq;
use heapless::FnvIndexMap;
#[allow(non_snake_case)]
use localization_core::*;
use nalgebra::Vector3;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;

#[test]
fn twr_3d_4anc_single_f32() {
    let r = StdRng::seed_from_u64(0);
    let location_gt: Vector3<f32> = Vector3::new(3.0, 0.0, 0.0);

    let location_a = Vector3::new(1.0, 0.0, 0.0);
    let location_b = Vector3::new(0.0, 1.0, 0.0);
    let location_c = Vector3::new(0.0, 0.0, 1.0);
    let location_d = Vector3::new(0.5, 0.0, 1.0);

    let mut locs: FnvIndexMap<u8, Vector3<f32>, 8> = FnvIndexMap::new();
    locs.insert(1, location_a).unwrap();
    locs.insert(2, location_b).unwrap();
    locs.insert(3, location_c).unwrap();
    locs.insert(4, location_d).unwrap();

    let mut trilateration_infos: FnvIndexMap<u8, f32, 8> = FnvIndexMap::new();
    for (node, loc) in &locs {
        let _ = trilateration_infos.insert(*node, (location_gt - loc).norm());
    }

    let mut solver = LocationSolver::new(&locs, 1e-8);
    let result = solver.trilateration(trilateration_infos, r).unwrap();

    assert_relative_eq!(location_gt, result, epsilon = 1e-3); // assert to 1 millimeter accuracy
}

#[test]
fn twr_3d_4anc_single_f64() {
    let r = StdRng::seed_from_u64(0);
    let location_gt = Vector3::new(3.0, 0.0, 0.0);

    let location_a = Vector3::new(1.0, 0.0, 0.0);
    let location_b = Vector3::new(0.0, 1.0, 0.0);
    let location_c = Vector3::new(0.0, 0.0, 1.0);
    let location_d = Vector3::new(0.5, 0.0, 1.0);

    let mut locs: FnvIndexMap<u8, Vector3<f64>, 8> = FnvIndexMap::new();
    locs.insert(1, location_a).unwrap();
    locs.insert(2, location_b).unwrap();
    locs.insert(3, location_c).unwrap();
    locs.insert(4, location_d).unwrap();

    let mut trilateration_infos: FnvIndexMap<u8, f64, 8> = FnvIndexMap::new();
    for (node, loc) in &locs {
        let _ = trilateration_infos.insert(*node, (location_gt - loc).norm());
    }

    let mut solver = LocationSolver::new(&locs, 1e-8);
    let result = solver.trilateration(trilateration_infos, r).unwrap();

    assert_relative_eq!(location_gt, result, epsilon = 1e-3); // assert to 1 millimeter accuracy
}

#[test]
fn twr_3d_4anc_multiple_random_f32() {
    let mut r = StdRng::seed_from_u64(0);
    let between = Uniform::from(-100.0..=100.0);

    let mut fail_count: usize = 0;
    let mut innacc_count: usize = 0;
    for _ in 0..1000 {
        let location_gt = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        let location_a = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );
        let location_b = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );
        let location_c = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );
        let location_d = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        let mut locs: FnvIndexMap<u8, Vector3<f32>, 8> = FnvIndexMap::new();
        locs.insert(1, location_a).unwrap();
        locs.insert(2, location_b).unwrap();
        locs.insert(3, location_c).unwrap();
        locs.insert(4, location_d).unwrap();

        let mut trilateration_infos: FnvIndexMap<u8, f32, 8> = FnvIndexMap::new();
        for (node, loc) in &locs {
            let _ = trilateration_infos.insert(*node, (location_gt.cast::<f64>() - loc.cast::<f64>()).norm() as f32);
        }

        let mut solver = LocationSolver::new(&locs, 1e-8);
        match solver.trilateration(trilateration_infos, r.clone()) {
            Ok(result) => {
                if (location_gt - result).norm() > 0.1 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert_eq!(innacc_count, 0);
    assert_eq!(fail_count, 0);
}

#[test]
fn twr_3d_4anc_multiple_random_f64() {
    let mut r = StdRng::seed_from_u64(0);
    let between = Uniform::from(-100.0..=100.0);

    let mut fail_count: usize = 0;
    let mut innacc_count: usize = 0;
    for _ in 0..1000 {
        let location_gt = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        let location_a = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );
        let location_b = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );
        let location_c = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );
        let location_d = Vector3::new(
            between.sample(&mut r),
            between.sample(&mut r),
            between.sample(&mut r),
        );

        let mut locs: FnvIndexMap<u8, Vector3<f64>, 8> = FnvIndexMap::new();
        locs.insert(1, location_a).unwrap();
        locs.insert(2, location_b).unwrap();
        locs.insert(3, location_c).unwrap();
        locs.insert(4, location_d).unwrap();

        let mut trilateration_infos: FnvIndexMap<u8, f64, 8> = FnvIndexMap::new();
        for (node, loc) in &locs {
            let _ = trilateration_infos.insert(*node, (location_gt - loc).norm());
        }

        let mut solver = LocationSolver::new(&locs, 1e-8);
        match solver.trilateration(trilateration_infos, r.clone()) {
            Ok(result) => {
                if (location_gt - result).norm() > 1e-3 {
                    // assert to 1 millimeter accuracy
                    innacc_count += 1;
                }
            }
            Err(_) => fail_count += 1,
        }
    }

    assert_eq!(innacc_count, 0);
    assert_eq!(fail_count, 0);
}
