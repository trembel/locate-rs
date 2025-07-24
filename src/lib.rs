#![cfg_attr(not(test), no_std)]

use heapless::{FnvIndexMap, Vec};
use nalgebra::{ComplexField, Matrix3, RealField, RowVector3, SMatrix, SVector, Unit, Vector3};
use num_traits::float::TotalOrder;
use rand::RngCore;

pub struct LocationSolver<'a, NODE, FLOAT, const MAXNNODES: usize> {
    known_locations: &'a FnvIndexMap<NODE, Vector3<FLOAT>, MAXNNODES>,
    solving_tolerance: FLOAT,
}

#[allow(non_snake_case)]
impl<'a, NODE, FLOAT, const MAXNNODES: usize> LocationSolver<'a, NODE, FLOAT, MAXNNODES>
where
    NODE: core::cmp::Eq,
    NODE: core::fmt::Debug,
    NODE: core::hash::Hash,
    NODE: Copy,
    FLOAT: num_traits::float::Float,
    FLOAT: RealField,
    FLOAT: simba::scalar::SubsetOf<f64>,
    FLOAT: TotalOrder,
{
    pub fn new(
        known_locations: &FnvIndexMap<NODE, Vector3<FLOAT>, MAXNNODES>,
        solving_tolerance: FLOAT,
    ) -> LocationSolver<NODE, FLOAT, MAXNNODES> {
        LocationSolver {
            known_locations,
            solving_tolerance,
        }
    }

    pub fn tdoa(
        &mut self,
        tdoa_distance_infos: FnvIndexMap<(NODE, NODE), FLOAT, MAXNNODES>,
        initial_guess: Option<Vector3<FLOAT>>,
    ) -> Result<Vector3<FLOAT>, ()> {
        if tdoa_distance_infos.len() < 3 || self.known_locations.len() < 4 {
            return Err(());
        }
        // If no initial guess is provided, initialize x at the center of the known anchor locations.
        let mut x = initial_guess.unwrap_or_else(|| {
            let num_known = self.known_locations.len();
            if num_known == 0 {
                Vector3::zeros()
            } else {
                let sum_of_positions: Vector3<FLOAT> = self.known_locations.values().sum();
                sum_of_positions / FLOAT::from_usize(num_known).unwrap()
            }
        });

        let lambda = FLOAT::from_f64(1e-3).unwrap(); // Levenberg-Marquardt damping factor
        let max_iterations: usize = 20;

        let len_tdoa_distance_infos = tdoa_distance_infos.len();

        for _ in 0..max_iterations {
            // Populate Jacobian matrix
            let mut residuals = SVector::<FLOAT, MAXNNODES>::zeros();
            let mut jacobian = SMatrix::<FLOAT, MAXNNODES, 3>::zeros();

            for (k, (&(i, j), &delta_d_ij)) in tdoa_distance_infos.iter().enumerate() {
                let Pi = self.known_locations.get(&i).unwrap();
                let Pj = self.known_locations.get(&j).unwrap();

                let vi = x - *Pi;
                let di = vi.norm();
                let vj = x - *Pj;
                let dj = vj.norm();

                residuals[k] = di - dj - delta_d_ij;

                // ∂r/∂x = (x - Pi)/‖x - Pi‖ - (x - Pj)/‖x - Pj‖
                let grad = vi / di - vj / dj;
                jacobian.row_mut(k).copy_from(&grad.transpose());
            }

            // Manually compute JᵀJ and Jᵀr to avoid allocation (or too many operations)
            let mut jtj = Matrix3::<FLOAT>::zeros(); // JᵀJ will be a 3x3 matrix
            let mut jtr = Vector3::<FLOAT>::zeros(); // Jᵀr will be a 3x1 vector

            for k in 0..len_tdoa_distance_infos {
                let row_k: RowVector3<FLOAT> = jacobian.row(k).into(); // Get the k-th row as a RowVector3 (1x3)
                let res_k: FLOAT = residuals[k]; // Get the k-th residual (scalar)

                // Accumulate JᵀJ: (3x1) * (1x3) = 3x3 matrix.
                // This multiplication is SIMD-optimized by nalgebra.
                jtj += row_k.transpose() * row_k;

                // Accumulate Jᵀr: (3x1) * scalar = 3x1 vector.
                // This multiplication is SIMD-optimized by nalgebra.
                jtr += row_k.transpose() * res_k;
            }

            // Solve Levenberg-Marquard iteration
            let lhs = jtj + Matrix3::identity() * lambda;
            let rhs = -jtr;

            // Solve (JᵀJ + λI) δ = -Jᵀr
            let delta = lhs.lu().solve(&rhs).ok_or(())?;

            x += delta;

            if delta.norm() < self.solving_tolerance {
                break;
            }
        }

        Ok(x)
    }

    pub fn trilateration_fast(
        &mut self,
        trilateration_infos: FnvIndexMap<NODE, FLOAT, MAXNNODES>,
        initial_guess: Option<Vector3<FLOAT>>,
    ) -> Result<Vector3<FLOAT>, ()> {
        if trilateration_infos.len() < 4 || self.known_locations.len() < 4 {
            return Err(());
        }

        // If no initial guess is provided, initialize x at the center of the known anchor locations.
        let mut x = initial_guess.unwrap_or_else(|| {
            let num_known = self.known_locations.len();
            if num_known == 0 {
                Vector3::zeros()
            } else {
                let sum_of_positions: Vector3<FLOAT> = self.known_locations.values().sum();
                sum_of_positions / FLOAT::from_usize(num_known).unwrap()
            }
        });

        let lambda = FLOAT::from_f64(1e-3).unwrap(); // Levenberg-Marquardt damping
        let max_iterations: usize = 50;

        let len_trilateration_infos = trilateration_infos.len();

        for _ in 0..max_iterations {
            // Prepare residuals and Jacobian
            let mut residuals = SVector::<FLOAT, MAXNNODES>::zeros();
            let mut jacobian = SMatrix::<FLOAT, MAXNNODES, 3>::zeros();

            for (k, (&node, &measured_distance)) in trilateration_infos.iter().enumerate() {
                let anchor = self.known_locations.get(&node).unwrap();
                let vec = x - *anchor;
                let dist = vec.norm();

                residuals[k] = dist - measured_distance;

                // ∂r/∂x = (x - anchor) / ‖x - anchor‖
                jacobian.row_mut(k).copy_from(&(vec / dist).transpose());
            }

            // Manually compute JᵀJ and Jᵀr to avoid allocation (or too many operations)
            let mut jtj = Matrix3::<FLOAT>::zeros(); // JᵀJ will be a 3x3 matrix
            let mut jtr = Vector3::<FLOAT>::zeros(); // Jᵀr will be a 3x1 vector

            for k in 0..len_trilateration_infos {
                let row_k: RowVector3<FLOAT> = jacobian.row(k).into(); // Get the k-th row as a RowVector3 (1x3)
                let res_k: FLOAT = residuals[k]; // Get the k-th residual (scalar)

                // Accumulate JᵀJ: (3x1) * (1x3) = 3x3 matrix.
                // This multiplication is SIMD-optimized by nalgebra.
                jtj += row_k.transpose() * row_k;

                // Accumulate Jᵀr: (3x1) * scalar = 3x1 vector.
                // This multiplication is SIMD-optimized by nalgebra.
                jtr += row_k.transpose() * res_k;
            }

            // Solve Levenberg-Marquard iteration
            let lhs = jtj + Matrix3::identity() * lambda;
            let rhs = -jtr;

            // Solve (JᵀJ + λI) δ = -Jᵀr
            let delta = lhs.lu().solve(&rhs).ok_or(())?;

            x += delta;

            if delta.norm() < self.solving_tolerance {
                break;
            }
        }

        Ok(x)
    }

    pub fn trilateration<RNG>(
        &mut self,
        trilateration_infos: FnvIndexMap<NODE, FLOAT, MAXNNODES>,
        rng: RNG,
    ) -> Result<Vector3<FLOAT>, ()>
    where
        RNG: RngCore,
    {
        // check for 3D sizes
        if trilateration_infos.len() < 4 || self.known_locations.len() < 4 {
            return Err(());
        }

        // From here follow 10.1109/icassp.2019.8683355, Chapter 1.4, 2.1, 2.2
        // get all location, distance^2, weight pairs (weight as in (6) of the paper)
        let mut data: Vec<(Vector3<FLOAT>, FLOAT, FLOAT), MAXNNODES> = Vec::new();
        for (node, distance) in trilateration_infos {
            let _ = data.push((
                self.known_locations[&node],
                ComplexField::powi(distance, 2),
                FLOAT::one() / FLOAT::from(4.0).unwrap() * ComplexField::powi(distance, 2),
            ));
        }

        // calculate t as translation value
        let mut w_sum = FLOAT::zero();
        let mut t: Vector3<FLOAT> = Vector3::zeros();
        for (p_i, _, w_i) in &data {
            w_sum += *w_i;
            t += p_i.map(|x| x * *w_i);
        }
        t = -(t / w_sum);

        // Compute matrix A and b
        let mut A: Matrix3<FLOAT> = Matrix3::zeros();
        let mut b: Vector3<FLOAT> = Vector3::zeros();

        for (p, d_pow2, mut w) in data.clone() {
            // normalize w
            w /= w_sum;

            // translate s
            let s: Vector3<FLOAT> = p + t;

            // Compute w*((s' * s) - d^2) which results in a scalar (1x1 matrix)
            let w_s_ts_minus_dpow2 = w * ((s.transpose() * s)[(0, 0)] - d_pow2);

            // Update matrix A (remember, s_ts is a 1x1 matrix, so extract its scalar value with `[(0, 0)]`)
            A += Matrix3::from_diagonal_element(w_s_ts_minus_dpow2)
                + (s * s.transpose()).map(|x| w * FLOAT::from(2.0).unwrap() * x);
            // Update vector b
            b += s.map(|x| x * -w_s_ts_minus_dpow2);
        }
        // Compute matrix D, containing eigenvalues of A
        let A_decomp = A.symmetric_eigen();
        let U = A_decomp.eigenvectors;
        let D = Matrix3::from_diagonal(&A_decomp.eigenvalues);

        // Transform b
        b = U.transpose() * b;

        // Generate M
        let mut M: SMatrix<FLOAT, 7, 7> = SMatrix::zeros(); // Initialize a 9x9 matrix of zeros
                                                            // Fill in the blocks
        M.view_mut((0, 0), (3, 3)).copy_from(&-D); // -D
        M.view_mut((0, 3), (3, 3))
            .copy_from(&-Matrix3::from_diagonal(&b));
        M.view_mut((3, 3), (3, 3)).copy_from(&-D); // -D
        M.view_mut((3, 6), (3, 1)).copy_from(&-b); // -b
        M.view_mut((6, 0), (1, 3))
            .copy_from(&RowVector3::repeat(FLOAT::one())); // identity

        // Get eigenvectors of M corresponding to largest real eigenvalue,
        // APPARENTLY: MAX EIGENVALUE CORRESPONDS TO THE OPTIMUM SOLUTION.. WHY? I DON'T KNOW!
        let mut x: SVector<FLOAT, 7> = self.get_ev(
            &M,
            M.complex_eigenvalues()
                .into_iter()
                .filter(|lambda| lambda.im == FLOAT::zero())
                .max_by(|a, b| a.re.partial_cmp(&b.re).unwrap())
                .unwrap()
                .re,
            rng,
        )?;
        // scale them by last value,
        x /= x[(6, 0)];
        // get elements 3:5
        let mut x = x.remove_row(6).remove_row(0).remove_row(0).remove_row(0);
        // transform to U*x - t
        x = U * x - t;

        // Return computed solution
        Ok(x)
    }

    #[allow(non_snake_case)]
    fn get_ev<RNG>(
        &mut self,
        A: &SMatrix<FLOAT, 7, 7>,
        lambda: FLOAT,
        mut rng: RNG,
    ) -> Result<SVector<FLOAT, 7>, ()>
    where
        RNG: RngCore,
    {
        // create 1e-3 deviation of lambda, s.t. matrix to be inversed is non-singular (works better with f64)
        let mu = lambda - FLOAT::from(1e-3).unwrap();

        // Generate matrix M
        let M = A - SMatrix::<FLOAT, 7, 7>::from_diagonal_element(mu);
        // Get LU transform to solve M^-1 * b
        let M = M.lu();

        let mut b = Unit::new_normalize(
            SVector::<FLOAT, 7>::zeros()
                .map(|_| FLOAT::from(rng.next_u32()).unwrap() / FLOAT::from(u32::MAX).unwrap()),
        );

        // Should converge pretty fast, anyway restrict iterations by 1000 - most of the time it will take less time
        for _ in 0..1000 {
            let b_new: Unit<SVector<FLOAT, 7>> = Unit::new_normalize(M.solve(&b).ok_or(())?);
            if (*b_new - *b).norm() < self.solving_tolerance {
                return Ok(*b);
            }
            b = b_new;
        }
        Ok(*b)
    }
}
