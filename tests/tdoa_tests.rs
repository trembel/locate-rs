use heapless::FnvIndexMap;
use localization_core::LocationSolver;
use nalgebra::Vector3;

#[test]
fn test_tdoa_from_example() {
    // 1. Define anchor positions
    let p1 = Vector3::new(0.0, 0.0, 0.0);
    let p2 = Vector3::new(10.0, 0.0, 0.0);
    let p3 = Vector3::new(0.0, 10.0, 10.0);
    let p4 = Vector3::new(0.0, 0.0, 10.0);

    let mut known_locations = FnvIndexMap::<&str, Vector3<f64>, 4>::new();
    known_locations.insert("p1", p1).unwrap();
    known_locations.insert("p2", p2).unwrap();
    known_locations.insert("p3", p3).unwrap();
    known_locations.insert("p4", p4).unwrap();

    // 2. Ground truth position
    let x_gt = Vector3::new(3.4, 4.5, 6.7);

    // 3. Compute true distances and TDoAs relative to anchor "p1"
    let r1 = (x_gt - p1).norm();
    let r2 = (x_gt - p2).norm();
    let r3 = (x_gt - p3).norm();
    let r4 = (x_gt - p4).norm();

    let r21 = r2 - r1;
    let r31 = r3 - r1;
    let r41 = r4 - r1;

    let mut tdoa_infos = FnvIndexMap::<(&str, &str), f64, 4>::new();
    tdoa_infos.insert(("p2", "p1"), r21).unwrap();
    tdoa_infos.insert(("p3", "p1"), r31).unwrap();
    tdoa_infos.insert(("p4", "p1"), r41).unwrap();

    // 4. Create solver and solve
    let solving_tolerance = 1e-6;
    let mut solver = LocationSolver::new(&known_locations, solving_tolerance);
    let result = solver.tdoa(tdoa_infos, None);

    // 5. Print and assert results
    assert!(result.is_ok());
    let x_est = result.unwrap();

    println!("Estimated position: {}", x_est.transpose());
    println!("Ground truth:       {}", x_gt.transpose());
    println!("Error:              {}", (x_est - x_gt).norm());

    // Check that the error is small
    assert!((x_est - x_gt).norm() < solving_tolerance);
}
