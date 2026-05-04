use ndarray::parallel::prelude::*;
use ndarray::Zip;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

#[pyfunction]
pub fn apply_igcr4_spin_combo6_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    q4_params: PyReadonlyArray2<f64>,
    norb: usize,
    alpha_occ: PyReadonlyArray2<u8>,
    beta_occ: PyReadonlyArray2<u8>,
) {
    let q4 = q4_params.as_array();
    let mut vec = vec.as_array_mut();
    let alpha_occ = alpha_occ.as_array();
    let beta_occ = beta_occ.as_array();

    let dim_b = vec.shape()[1];

    Zip::indexed(vec.rows_mut())
        .and(alpha_occ.rows())
        .par_for_each(|_, mut row, occ_a| {
            for ib in 0..dim_b {
                let occ_b = beta_occ.row(ib);
                let mut phi = 0.0f64;
                let mut idx = 0usize;

                for p in 0..norb {
                    let ap = occ_a[p] as f64;
                    let bp = occ_b[p] as f64;
                    for q in 0..=p {
                        let aq = occ_a[q] as f64;
                        let bq = occ_b[q] as f64;
                        for r in 0..=q {
                            let ar = occ_a[r] as f64;
                            let br = occ_b[r] as f64;
                            for sidx in 0..=r {
                                let a_s = occ_a[sidx] as f64;
                                let b_s = occ_b[sidx] as f64;
                                phi += q4[(idx, 0)] * ap * aq * br * b_s;
                                phi += q4[(idx, 1)] * ap * bq * ar * b_s;
                                phi += q4[(idx, 2)] * ap * bq * br * a_s;
                                phi += q4[(idx, 3)] * bp * aq * ar * b_s;
                                phi += q4[(idx, 4)] * bp * aq * br * a_s;
                                phi += q4[(idx, 5)] * bp * bq * ar * a_s;
                                idx += 1;
                            }
                        }
                    }
                }

                let (s, c) = phi.sin_cos();
                row[ib] *= Complex64::new(c, s);
            }
        });
}
