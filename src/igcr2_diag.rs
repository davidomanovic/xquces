use ndarray::Array2;
use ndarray::Zip;
use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

/// Apply the irreducible GCR-2 diagonal correlator in-place.
///
/// Implements:
///   U_J(β, γ) = exp[i Σ_{p=1}^{norb-1} β_p D_{p+1} + i Σ_{p<q} γ_{pq} N_p N_q]
///
/// where:
///   - D_p = n_{pα} n_{pβ}  (double occupancy on spatial orbital p)
///   - N_p = n_{pα} + n_{pβ} (total occupation of spatial orbital p)
///   - β has length (norb - 1), with β[k] corresponding to orbital p = k+1
///     (orbital 0 is the reference whose D_0 is eliminated)
///   - γ_exp is upper-triangular (norb × norb) with pre-exponentiated pair phases
///
/// Arguments:
///   vec:       statevector in (dim_alpha, dim_beta) shape, modified in-place
///   double_exp: exp(i β_p) for p = 1, ..., norb-1.  Length = norb - 1.
///   pair_exp:   exp(i γ_{pq}) for 0 ≤ p < q ≤ norb-1.  Shape = (norb, norb), upper tri.
///   norb:       number of spatial orbitals
///   occupations_a: alpha occupation lists, shape (dim_alpha, n_alpha)
///   occupations_b: beta  occupation lists, shape (dim_beta,  n_beta)
#[pyfunction]
pub fn apply_igcr2_diag_in_place(
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

    // Build full occupation masks
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

    // Apply phases in-place
    Zip::indexed(vec.rows_mut())
        .and(alpha_occ.rows())
        .par_for_each(|_, mut row, occ_a| {
            for ib in 0..dim_b {
                let occ_b = beta_occ.row(ib);
                let mut phase = Complex64::new(1.0, 0.0);

                // Double-occupancy phases: skip orbital 0 (reference)
                for p in 1..norb {
                    if occ_a[p] == 1 && occ_b[p] == 1 {
                        phase *= double_exp[p - 1]; // β index offset by 1
                    }
                }

                // Pair density phases: N_p * N_q for all 0 ≤ p < q
                for p in 0..norb {
                    let count_p = occ_a[p] + occ_b[p];
                    if count_p == 0 {
                        continue;
                    }
                    for q in (p + 1)..norb {
                        let count_q = occ_a[q] + occ_b[q];
                        if count_q == 0 {
                            continue;
                        }
                        // N_p * N_q ∈ {1, 2, 4}, so we need pair_exp^(count_p * count_q)
                        let prod = count_p * count_q;
                        let base = pair_exp[(p, q)];
                        let factor = match prod {
                            1 => base,
                            2 => base * base,
                            4 => {
                                let b2 = base * base;
                                b2 * b2
                            }
                            _ => unreachable!(),
                        };
                        phase *= factor;
                    }
                }

                row[ib] *= phase;
            }
        });
}