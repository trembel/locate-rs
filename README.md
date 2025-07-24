# locate-rs

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Crates.io](https://img.shields.io/crates/v/locate-rs.svg)](https://crates.io/crates/locate-rs)
[![Docs.rs](https://docs.rs/locate-rs/badge.svg)](https://docs.rs/locate-rs)

A `no_std` Rust library for 3D localization using TDOA and Trilateration. It provides fast iterative solvers (Levenberg-Marquardt) and a slower closed-form, eigenvector-based solution (trilateration only) for high-accuracy positioning.

This library is designed for resource-constrained environments where `alloc` is not available.

If you use *locate-rs* in an academic or industrial context, please cite the following publication:

```bibtex
@misc{cortesi25_wakeloc,
  title          = {WakeLoc: An Ultra-Low Power, Accurate and Scalable On-Demand RTLS using Wake-Up Radios},
  author         = {Silvano Cortesi and Christian Vogt and Michele Magno},
  year           = 2025,
  doi            = {10.48550/arXiv.2504.20545},
  url            = {https://doi.org/10.48550/arXiv.2504.20545},
  eprint         = {2504.20545},
  archiveprefix  = {arXiv},
  primaryclass   = {cs.NI}
}
```

## Features

- **`no_std` compatible:** Suitable for embedded systems and other environments without a standard library.
- **3D Localization:** Accurately computes 3D positions.
- **TDoA and Trilateration:** Supports both Time Difference of Arrival and Trilateration methods.
- **Multiple Solvers:**
    - **Iterative Solver:** A fast Levenberg-Marquardt solver for both TDoA and trilateration.
    - **Eigenvalue-based Solver:** A closed-form, non-iterative solver for trilateration, based on the paper "Optimal Trilateration is an Eigenvalue Problem".
- **Generic:** Flexible and adaptable to different numeric types and node identifiers.

## Getting Started

Add `locate-rs` to your `Cargo.toml`:

```toml
[dependencies]
locate-rs = "1.0"
nalgebra = { version = "0.33.2", default-features = false, features=["libm"] }
heapless = "0.8.0"
```

## Usage

Here's a basic example of how to use `LocationSolver` for trilateration:

```rust
use locate_rs::{LocationSolver, LocationError};
use nalgebra::Vector3;
use heapless::FnvIndexMap;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() -> Result<(), LocationError> {
    // 1. Define anchor positions
    let mut known_locations = FnvIndexMap::<&str, Vector3<f64>, 4>::new();
    known_locations.insert("A", Vector3::new(0.0, 0.0, 0.0)).unwrap();
    known_locations.insert("B", Vector3::new(10.0, 0.0, 0.0)).unwrap();
    known_locations.insert("C", Vector3::new(0.0, 10.0, 10.0)).unwrap();
    known_locations.insert("D", Vector3::new(0.0, 0.0, 10.0)).unwrap();

    // 2. Ground truth position
    let x_gt = Vector3::new(3.4, 4.5, 6.7);

    // 3. Create distance map
    let mut trilateration_infos = FnvIndexMap::<&str, f64, 4>::new();
    trilateration_infos.insert("A", 8.758).unwrap();
    trilateration_infos.insert("B", 10.426).unwrap();
    trilateration_infos.insert("C", 7.259).unwrap();
    trilateration_infos.insert("D", 6.535).unwrap();

    // 4. Create solver and solve
    let mut solver = LocationSolver::new(&known_locations, 1e-6);

    // using iterative solver
    let location_fast = solver.trilateration_fast(trilateration_infos.clone(), None)?; // No initial guess of position

    // using eigenvalue optimal solver
    let mut rng = SmallRng::from_seed([0; 32]);
    let location_eigen = solver.trilateration(trilateration_infos, &mut rng)?;

    println!("Trilateration (fast) solution: {:?}\r\nTrilateration (eigenvalue) solution: {:?}\r\nGroundtruth: {:?}", location_fast, location_eigen, x_gt);
    Ok(())
}
```

For TDoA, you would provide distance differences instead:

```rust
use locate_rs::{LocationSolver, LocationError};
use nalgebra::Vector3;
use heapless::FnvIndexMap;

fn main() -> Result<(), LocationError> {
    // 1. Define anchor positions
    let mut known_locations = FnvIndexMap::<&str, Vector3<f64>, 4>::new();
    known_locations.insert("A", Vector3::new(0.0, 0.0, 0.0)).unwrap();
    known_locations.insert("B", Vector3::new(10.0, 0.0, 0.0)).unwrap();
    known_locations.insert("C", Vector3::new(0.0, 10.0, 10.0)).unwrap();
    known_locations.insert("D", Vector3::new(0.0, 0.0, 10.0)).unwrap();

    // 2. Ground truth position
    let x_gt = Vector3::new(3.4, 4.5, 6.7);

    // 3. Create TDoAs relative to anchor "A"
    let mut tdoa_infos = FnvIndexMap::<(&str, &str), f64, 4>::new();
    tdoa_infos.insert(("B", "A"), 1.668).unwrap(); // d_B - d_A
    tdoa_infos.insert(("C", "A"), -1.498).unwrap(); // d_C - d_A
    tdoa_infos.insert(("D", "A"), -2.223).unwrap(); // d_D - d_A

    // 4. Create solver and solve
    let mut solver = LocationSolver::new(&known_locations, 1e-6);
    let location_tdoa = solver.tdoa(tdoa_infos, None)?; // No initial guess of position

    println!("TDoA solution: {:?},\r\nGroundtruth: {:?}", location_tdoa, x_gt);
    Ok(())
}
```

## Running Tests

To run the tests for this library, use the following command:

```bash
cargo test
```

## Running Benchmarks

The repository includes benchmarks using `criterion`. To run them:

```bash
cargo bench
```

## License

This project is licensed under the LGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
