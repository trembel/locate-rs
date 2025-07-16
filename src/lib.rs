//#![cfg_attr(not(test), no_std)]

use heapless::{FnvIndexMap, Vec};
use nalgebra::{ComplexField, Matrix3, RealField, RowVector3, SMatrix, SVector, Unit, Vector3};
use num_traits::float::TotalOrder;
use rand::RngCore;

pub struct LocationSolver<'a, NODE, FLOAT, const NNODES: usize> {
    known_locations: &'a FnvIndexMap<NODE, Vector3<FLOAT>, NNODES>,
    solving_tolerance: FLOAT,
}

#[allow(non_snake_case)]
impl<'a, NODE, FLOAT, const NNODES: usize> LocationSolver<'a, NODE, FLOAT, NNODES>
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
        known_locations: &FnvIndexMap<NODE, Vector3<FLOAT>, NNODES>,
        solving_tolerance: FLOAT,
    ) -> LocationSolver<NODE, FLOAT, NNODES> {
        LocationSolver {
            known_locations,
            solving_tolerance,
        }
    }

    pub fn tdoa<const INFOSCAPACITY: usize>(
        &mut self,
        tdoa_infos: FnvIndexMap<(NODE, NODE), FLOAT, INFOSCAPACITY>,
        initial_guess: Option<Vector3<FLOAT>>,
    ) -> Result<Vector3<FLOAT>, ()> {
        // Internal macro for compile-time dispatch
        macro_rules! dispatch_tdoa {
            ($solver:expr, $infos:expr, $guess:expr, $cap:expr, [$($npairs:literal),+]) => {
                match $infos.len() {
                    $(
                          $npairs => $solver.internal_tdoa::<$npairs, $cap>($infos, $guess),
                    )+
                    _ => Err(()),
                }
            };
        }

        dispatch_tdoa!(
            self,
            tdoa_infos,
            initial_guess,
            INFOSCAPACITY,
            [3, 6, 10, 15, 21]
        )
    }

    pub fn internal_tdoa<const NPAIRS: usize, const INFOSCAPACITY: usize>(
        &mut self,
        tdoa_infos: FnvIndexMap<(NODE, NODE), FLOAT, INFOSCAPACITY>,
        initial_guess: Option<Vector3<FLOAT>>,
    ) -> Result<Vector3<FLOAT>, ()> {
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

        let lambda = FLOAT::from_f64(1e-3).unwrap(); // Levenberg damping factor
        let max_iterations = 20;

        for it in 0..max_iterations {
            // Populate Jacobian matrix
            let mut residuals = SVector::<FLOAT, NPAIRS>::zeros();
            let mut jacobian = SMatrix::<FLOAT, NPAIRS, 3>::zeros();

            for (k, (&(i, j), &delta_d_ij)) in tdoa_infos.iter().enumerate() {
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

            // Solve Levenberg-Marquard iteration
            let jt = jacobian.transpose();
            let jtj = jt * jacobian;
            let lhs = jtj + Matrix3::identity() * lambda;
            let rhs = -jt * residuals;

            // Solve (JᵀJ + λI) δ = -Jᵀr
            let delta = lhs.lu().solve(&rhs).ok_or(())?;

            x += delta;

            if delta.norm() < self.solving_tolerance {
                println!("{:}", it);
                break;
            }
        }

        Ok(x)
    }

    pub fn trilateration<RNG>(
        &mut self,
        trilateration_infos: FnvIndexMap<NODE, FLOAT, NNODES>,
        rng: RNG,
    ) -> Result<SVector<FLOAT, 3>, ()>
    where
        RNG: RngCore,
    {
        // check for 3D sizes
        if trilateration_infos.len() < 4 || self.known_locations.len() < 4 {
            return Err(());
        }

        // From here follow 10.1109/icassp.2019.8683355, Chapter 1.4, 2.1, 2.2
        // get all location, distance^2, weight pairs (weight as in (6) of the paper)
        let mut data: Vec<(Vector3<FLOAT>, FLOAT, FLOAT), NNODES> = Vec::new();
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
        // create 1e-6 deviation of lambda, s.t. matrix to be inversed is non-singular (needs f64)
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
        return Ok(*b);
    }
}
