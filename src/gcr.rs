use ndarray::Array2;
use ndarray::Zip;
use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

#[pyfunction]
pub fn apply_gcr_spin_balanced_reduced_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    mu_exp: PyReadonlyArray1<Complex64>,
    nu_exp: PyReadonlyArray1<Complex64>,
    alpha_exp: PyReadonlyArray2<Complex64>,
    beta_exp: PyReadonlyArray2<Complex64>,
    norb: usize,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
) {
    let mu_exp = mu_exp.as_array();
    let nu_exp = nu_exp.as_array();
    let alpha_exp = alpha_exp.as_array();
    let beta_exp = beta_exp.as_array();
    let mut vec = vec.as_array_mut();
    let occupations_a = occupations_a.as_array();
    let occupations_b = occupations_b.as_array();

    let dim_a = vec.shape()[0];
    let dim_b = vec.shape()[1];
    let n_alpha = occupations_a.shape()[1];
    let n_beta = occupations_b.shape()[1];

    let mut alpha_occ = Array2::<usize>::zeros((dim_a, norb));
    let mut beta_occ = Array2::<usize>::zeros((dim_b, norb));

    Zip::from(alpha_occ.rows_mut())
        .and(occupations_a.rows())
        .par_for_each(|mut row, orbs| {
            for j in 0..n_alpha {
                row[orbs[j]] = 1;
            }
        });

    Zip::from(beta_occ.rows_mut())
        .and(occupations_b.rows())
        .par_for_each(|mut row, orbs| {
            for j in 0..n_beta {
                row[orbs[j]] = 1;
            }
        });

    Zip::indexed(vec.rows_mut())
        .and(alpha_occ.rows())
        .par_for_each(|_, mut row, occ_a| {
            for ib in 0..dim_b {
                let occ_b = beta_occ.row(ib);
                let mut phase = Complex64::new(1.0, 0.0);

                for p in 0..norb {
                    if occ_a[p] == 1 {
                        phase *= mu_exp[p];
                    }
                    if occ_b[p] == 1 {
                        phase *= mu_exp[p];
                    }
                    if occ_a[p] == 1 && occ_b[p] == 1 {
                        phase *= nu_exp[p];
                    }
                }

                for p in 0..norb {
                    for q in (p + 1)..norb {
                        if occ_a[p] == 1 && occ_a[q] == 1 {
                            phase *= alpha_exp[(p, q)];
                        }
                        if occ_b[p] == 1 && occ_b[q] == 1 {
                            phase *= alpha_exp[(p, q)];
                        }
                        if occ_a[p] == 1 && occ_b[q] == 1 {
                            phase *= beta_exp[(p, q)];
                        }
                        if occ_b[p] == 1 && occ_a[q] == 1 {
                            phase *= beta_exp[(p, q)];
                        }
                    }
                }

                row[ib] *= phase;
            }
        });
}