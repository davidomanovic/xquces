use ndarray::Array2;
use ndarray::Zip;
use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

fn pow_small(base: Complex64, exp: usize) -> Complex64 {
    let mut out = Complex64::new(1.0, 0.0);
    for _ in 0..exp {
        out *= base;
    }
    out
}

#[pyfunction]
pub fn apply_ucj_spin_restricted_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    double_exp: PyReadonlyArray1<Complex64>,
    pair_exp: PyReadonlyArray2<Complex64>,
    norb: usize,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
) {
    let double_exp = double_exp.as_array();
    let pair_exp = pair_exp.as_array();
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
                    let count_p = occ_a[p] + occ_b[p];
                    if count_p == 2 {
                        phase *= double_exp[p];
                    }
                    if count_p == 0 {
                        continue;
                    }
                    for q in (p + 1)..norb {
                        let count_q = occ_a[q] + occ_b[q];
                        if count_q == 0 {
                            continue;
                        }
                        phase *= pow_small(pair_exp[(p, q)], count_p * count_q);
                    }
                }

                row[ib] *= phase;
            }
        });
}

#[pyfunction]
pub fn apply_ucj_spin_balanced_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    same_exp: PyReadonlyArray2<Complex64>,
    mixed_exp: PyReadonlyArray2<Complex64>,
    same_diag_exp: PyReadonlyArray1<Complex64>,
    mixed_diag_exp: PyReadonlyArray1<Complex64>,
    norb: usize,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
) {
    let same_exp = same_exp.as_array();
    let mixed_exp = mixed_exp.as_array();
    let same_diag_exp = same_diag_exp.as_array();
    let mixed_diag_exp = mixed_diag_exp.as_array();
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
                        phase *= same_diag_exp[p];
                    }
                    if occ_b[p] == 1 {
                        phase *= same_diag_exp[p];
                    }
                    if occ_a[p] == 1 && occ_b[p] == 1 {
                        phase *= mixed_diag_exp[p];
                    }
                }

                for p in 0..norb {
                    for q in (p + 1)..norb {
                        if occ_a[p] == 1 && occ_a[q] == 1 {
                            phase *= same_exp[(p, q)];
                        }
                        if occ_b[p] == 1 && occ_b[q] == 1 {
                            phase *= same_exp[(p, q)];
                        }
                        if occ_a[p] == 1 && occ_b[q] == 1 {
                            phase *= mixed_exp[(p, q)];
                        }
                        if occ_b[p] == 1 && occ_a[q] == 1 {
                            phase *= mixed_exp[(p, q)];
                        }
                    }
                }

                row[ib] *= phase;
            }
        });
}

#[pyfunction]
pub fn apply_igcr3_spin_restricted_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    double_exp: PyReadonlyArray1<Complex64>,
    pair_exp: PyReadonlyArray2<Complex64>,
    tau_exp: PyReadonlyArray2<Complex64>,
    omega_exp: PyReadonlyArray1<Complex64>,
    norb: usize,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
) {
    let double_exp = double_exp.as_array();
    let pair_exp = pair_exp.as_array();
    let tau_exp = tau_exp.as_array();
    let omega_exp = omega_exp.as_array();
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
                    let count_p = occ_a[p] + occ_b[p];
                    if count_p == 0 {
                        continue;
                    }
                    if count_p == 2 {
                        phase *= double_exp[p];
                    }

                    for q in (p + 1)..norb {
                        let count_q = occ_a[q] + occ_b[q];
                        if count_q == 0 {
                            continue;
                        }
                        phase *= pow_small(pair_exp[(p, q)], count_p * count_q);
                    }
                }

                for p in 0..norb {
                    let count_p = occ_a[p] + occ_b[p];
                    if count_p != 2 {
                        continue;
                    }
                    for q in 0..norb {
                        if p == q {
                            continue;
                        }
                        let count_q = occ_a[q] + occ_b[q];
                        if count_q != 0 {
                            phase *= pow_small(tau_exp[(p, q)], count_q);
                        }
                    }
                }

                let mut omega_idx = 0;
                for p in 0..norb {
                    let count_p = occ_a[p] + occ_b[p];
                    for q in (p + 1)..norb {
                        let count_q = occ_a[q] + occ_b[q];
                        for r in (q + 1)..norb {
                            let count_r = occ_a[r] + occ_b[r];
                            let exponent = count_p * count_q * count_r;
                            if exponent != 0 {
                                phase *= pow_small(omega_exp[omega_idx], exponent);
                            }
                            omega_idx += 1;
                        }
                    }
                }

                row[ib] *= phase;
            }
        });
}
