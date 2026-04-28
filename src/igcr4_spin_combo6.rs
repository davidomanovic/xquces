use ndarray::parallel::prelude::*;
use ndarray::ArrayView1;
use ndarray::Zip;
use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn dot_occ(coeff: &[f64], occ: ArrayView1<u8>) -> f64 {
    let mut out = 0.0f64;
    for k in 0..coeff.len() {
        out += coeff[k] * occ[k] as f64;
    }
    out
}

#[pyfunction]
pub fn apply_igcr4_spin_combo6_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    double_params: PyReadonlyArray1<f64>,
    pair_params: PyReadonlyArray2<f64>,
    tau_params: PyReadonlyArray2<f64>,
    omega_params: PyReadonlyArray1<f64>,
    eta_params: PyReadonlyArray1<f64>,
    rho_params: PyReadonlyArray1<f64>,
    sigma6_params: PyReadonlyArray2<f64>,
    norb: usize,
    alpha_occ: PyReadonlyArray2<u8>,
    beta_occ: PyReadonlyArray2<u8>,
) {
    let lam = double_params.as_array();
    let pair = pair_params.as_array();
    let tau = tau_params.as_array();
    let omega = omega_params.as_array();
    let eta = eta_params.as_array();
    let rho = rho_params.as_array();
    let sigma6 = sigma6_params.as_array();
    let mut vec = vec.as_array_mut();
    let alpha_occ = alpha_occ.as_array();
    let beta_occ = beta_occ.as_array();

    let dim_a = vec.shape()[0];
    let dim_b = vec.shape()[1];

    let mut ps_a = vec![0.0f64; dim_a];
    let mut ps_b = vec![0.0f64; dim_b];
    let mut ooo_a = vec![0.0f64; dim_a];
    let mut ooo_b = vec![0.0f64; dim_b];
    let mut ca = vec![0.0f64; dim_a * norb];
    let mut cb = vec![0.0f64; dim_b * norb];

    ps_a
        .par_iter_mut()
        .zip(ooo_a.par_iter_mut())
        .zip(ca.par_chunks_mut(norb))
        .enumerate()
        .for_each(|(ia, ((ps, ooo), ca_row))| {
            let occ = alpha_occ.row(ia);
            for k in 0..norb {
                ca_row[k] += lam[k] * occ[k] as f64;
            }
            for p in 0..norb {
                if occ[p] == 0 {
                    continue;
                }
                for q in (p + 1)..norb {
                    let j = pair[(p, q)];
                    if j == 0.0 {
                        continue;
                    }
                    if occ[q] != 0 {
                        *ps += j;
                    }
                    ca_row[q] += j;
                }
                for q in 0..p {
                    let j = pair[(q, p)];
                    if j != 0.0 {
                        ca_row[q] += j;
                    }
                }
            }
            for p in 0..norb {
                if occ[p] == 0 {
                    continue;
                }
                let mut dot = 0.0f64;
                for q in 0..norb {
                    dot += tau[(p, q)] * occ[q] as f64;
                }
                ca_row[p] += dot;
            }
            let mut omega_idx = 0usize;
            for p in 0..norb {
                for q in (p + 1)..norb {
                    for r in (q + 1)..norb {
                        let w = omega[omega_idx];
                        omega_idx += 1;
                        if w == 0.0 {
                            continue;
                        }
                        let ap = occ[p] as f64;
                        let aq = occ[q] as f64;
                        let ar = occ[r] as f64;
                        *ooo += w * ap * aq * ar;
                        ca_row[r] += w * ap * aq;
                        ca_row[q] += w * ap * ar;
                        ca_row[p] += w * aq * ar;
                    }
                }
            }
        });

    ps_b
        .par_iter_mut()
        .zip(ooo_b.par_iter_mut())
        .zip(cb.par_chunks_mut(norb))
        .enumerate()
        .for_each(|(ib, ((ps, ooo), cb_row))| {
            let occ = beta_occ.row(ib);
            for p in 0..norb {
                if occ[p] == 0 {
                    continue;
                }
                for q in (p + 1)..norb {
                    let j = pair[(p, q)];
                    if j != 0.0 && occ[q] != 0 {
                        *ps += j;
                    }
                }
            }
            for p in 0..norb {
                if occ[p] == 0 {
                    continue;
                }
                let mut dot = 0.0f64;
                for q in 0..norb {
                    dot += tau[(p, q)] * occ[q] as f64;
                }
                cb_row[p] += dot;
            }
            let mut omega_idx = 0usize;
            for p in 0..norb {
                for q in (p + 1)..norb {
                    for r in (q + 1)..norb {
                        let w = omega[omega_idx];
                        omega_idx += 1;
                        if w == 0.0 {
                            continue;
                        }
                        let ap = occ[p] as f64;
                        let aq = occ[q] as f64;
                        let ar = occ[r] as f64;
                        *ooo += w * ap * aq * ar;
                        cb_row[r] += w * ap * aq;
                        cb_row[q] += w * ap * ar;
                        cb_row[p] += w * aq * ar;
                    }
                }
            }
        });

    Zip::indexed(vec.rows_mut())
        .and(alpha_occ.rows())
        .par_for_each(|ia, mut row, occ_a| {
            let ca_row = &ca[ia * norb..(ia + 1) * norb];
            let scalar_a = ps_a[ia] + ooo_a[ia];
            for ib in 0..dim_b {
                let occ_b = beta_occ.row(ib);
                let cb_row = &cb[ib * norb..(ib + 1) * norb];
                let mut phi = scalar_a + ps_b[ib] + ooo_b[ib];
                phi += dot_occ(ca_row, occ_b);
                phi += dot_occ(cb_row, occ_a);

                let mut eta_idx = 0usize;
                for p in 0..norb {
                    let dp = occ_a[p] != 0 && occ_b[p] != 0;
                    for q in (p + 1)..norb {
                        if dp && occ_a[q] != 0 && occ_b[q] != 0 {
                            phi += eta[eta_idx];
                        }
                        eta_idx += 1;
                    }
                }

                let mut rho_idx = 0usize;
                for p in 0..norb {
                    let dp = occ_a[p] != 0 && occ_b[p] != 0;
                    for q in 0..norb {
                        if q == p {
                            continue;
                        }
                        let count_q = occ_a[q] as f64 + occ_b[q] as f64;
                        for r in (q + 1)..norb {
                            if r == p {
                                continue;
                            }
                            if dp {
                                let count_r = occ_a[r] as f64 + occ_b[r] as f64;
                                phi += rho[rho_idx] * count_q * count_r;
                            }
                            rho_idx += 1;
                        }
                    }
                }

                let mut sigma_idx = 0usize;
                for p in 0..norb {
                    let ap = occ_a[p] as f64;
                    let bp = occ_b[p] as f64;
                    for q in (p + 1)..norb {
                        let aq = occ_a[q] as f64;
                        let bq = occ_b[q] as f64;
                        for r in (q + 1)..norb {
                            let ar = occ_a[r] as f64;
                            let br = occ_b[r] as f64;
                            for s in (r + 1)..norb {
                                let ass = occ_a[s] as f64;
                                let bs = occ_b[s] as f64;
                                phi += sigma6[(sigma_idx, 0)] * ap * aq * br * bs;
                                phi += sigma6[(sigma_idx, 1)] * ap * bq * ar * bs;
                                phi += sigma6[(sigma_idx, 2)] * ap * bq * br * ass;
                                phi += sigma6[(sigma_idx, 3)] * bp * aq * ar * bs;
                                phi += sigma6[(sigma_idx, 4)] * bp * aq * br * ass;
                                phi += sigma6[(sigma_idx, 5)] * bp * bq * ar * ass;
                                sigma_idx += 1;
                            }
                        }
                    }
                }

                let (s, c) = phi.sin_cos();
                row[ib] *= Complex64::new(c, s);
            }
        });
}
