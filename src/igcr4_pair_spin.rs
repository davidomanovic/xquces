use std::collections::HashMap;

use ndarray::ArrayView2;
use num_complex::Complex64;
use numpy::{PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn bits_from_occ(occ: ArrayView2<'_, u8>, norb: usize) -> PyResult<Vec<u64>> {
    if norb >= 63 {
        return Err(PyValueError::new_err(
            "pair-spin quartic Rust kernel supports norb < 63",
        ));
    }
    let mut out = Vec::with_capacity(occ.shape()[0]);
    for row in occ.rows() {
        if row.len() != norb {
            return Err(PyValueError::new_err("occupation array has wrong width"));
        }
        let mut bits = 0u64;
        for p in 0..norb {
            if row[p] != 0 {
                bits |= 1u64 << p;
            }
        }
        out.push(bits);
    }
    Ok(out)
}

#[inline]
fn parity_sign(n: u32) -> f64 {
    if n & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn replace_orbital(bits: u64, old: usize, new: usize) -> Option<(u64, f64)> {
    let old_mask = 1u64 << old;
    let new_mask = 1u64 << new;
    if bits & old_mask == 0 || bits & new_mask != 0 {
        return None;
    }
    let below_old = bits & (old_mask - 1);
    let sign_annihilate = parity_sign(below_old.count_ones());
    let removed = bits ^ old_mask;
    let below_new = removed & (new_mask - 1);
    let sign_create = parity_sign(below_new.count_ones());
    Some((removed | new_mask, sign_annihilate * sign_create))
}

fn build_index_map(bits: &[u64]) -> HashMap<u64, usize> {
    bits.iter()
        .copied()
        .enumerate()
        .map(|(idx, det)| (det, idx))
        .collect()
}

fn validate_spin_pairs(spin_pairs: ArrayView2<'_, usize>, norb: usize) -> PyResult<()> {
    if spin_pairs.shape()[1] != 2 {
        return Err(PyValueError::new_err("spin_pairs must have shape (n_pairs, 2)"));
    }
    let mut used = vec![false; norb];
    for k in 0..spin_pairs.shape()[0] {
        let p = spin_pairs[(k, 0)];
        let q = spin_pairs[(k, 1)];
        if p >= norb || q >= norb {
            return Err(PyValueError::new_err("spin_pairs index out of bounds"));
        }
        if p == q {
            return Err(PyValueError::new_err("spin_pairs entries must be distinct"));
        }
        if used[p] || used[q] {
            return Err(PyValueError::new_err("spin_pairs must be disjoint"));
        }
        used[p] = true;
        used[q] = true;
    }
    Ok(())
}

fn apply_pair_basis_transform(
    state: &mut [Complex64],
    alpha_bits: &[u64],
    beta_bits: &[u64],
    dim_b: usize,
    spin_pairs: ArrayView2<'_, usize>,
) -> PyResult<()> {
    let map_a = build_index_map(alpha_bits);
    let map_b = build_index_map(beta_bits);
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    for k in 0..spin_pairs.shape()[0] {
        let p = spin_pairs[(k, 0)];
        let q = spin_pairs[(k, 1)];
        let p_mask = 1u64 << p;
        let q_mask = 1u64 << q;

        for ia in 0..alpha_bits.len() {
            let alpha = alpha_bits[ia];
            if alpha & p_mask == 0 || alpha & q_mask != 0 {
                continue;
            }
            let (new_alpha, sign_a) = replace_orbital(alpha, p, q).unwrap();
            let target_ia = *map_a
                .get(&new_alpha)
                .ok_or_else(|| PyValueError::new_err("alpha target determinant missing"))?;

            for ib in 0..beta_bits.len() {
                let beta = beta_bits[ib];
                if beta & q_mask == 0 || beta & p_mask != 0 {
                    continue;
                }
                let (new_beta, sign_b) = replace_orbital(beta, q, p).unwrap();
                let target_ib = *map_b
                    .get(&new_beta)
                    .ok_or_else(|| PyValueError::new_err("beta target determinant missing"))?;
                let source = ia * dim_b + ib;
                let target = target_ia * dim_b + target_ib;
                let sign = sign_a * sign_b;
                let source_value = state[source];
                let target_value = state[target];
                state[source] = (source_value + target_value * sign) * inv_sqrt2;
                state[target] = (source_value * sign - target_value) * inv_sqrt2;
            }
        }
    }
    Ok(())
}

#[inline]
fn pair_label(alpha: u64, beta: u64, p: usize, q: usize) -> u8 {
    let ap = ((alpha >> p) & 1) as u8;
    let aq = ((alpha >> q) & 1) as u8;
    let bp = ((beta >> p) & 1) as u8;
    let bq = ((beta >> q) & 1) as u8;
    if ap + bp != 1 || aq + bq != 1 {
        return 0;
    }
    if ap == 0 && aq == 1 && bp == 1 && bq == 0 {
        1
    } else {
        2
    }
}

fn apply_pair_spin_phases(
    state: &mut [Complex64],
    alpha_bits: &[u64],
    beta_bits: &[u64],
    dim_b: usize,
    spin_pairs: ArrayView2<'_, usize>,
    theta_singlet: ArrayView2<'_, f64>,
    theta_triplet: ArrayView2<'_, f64>,
) -> PyResult<()> {
    let n_pairs = spin_pairs.shape()[0];
    if theta_singlet.shape() != [n_pairs, n_pairs] {
        return Err(PyValueError::new_err(
            "theta_singlet must have shape (n_pairs, n_pairs)",
        ));
    }
    if theta_triplet.shape() != [n_pairs, n_pairs] {
        return Err(PyValueError::new_err(
            "theta_triplet must have shape (n_pairs, n_pairs)",
        ));
    }
    let mut labels = vec![0u8; n_pairs];

    for ia in 0..alpha_bits.len() {
        let alpha = alpha_bits[ia];
        for ib in 0..beta_bits.len() {
            let beta = beta_bits[ib];
            for k in 0..n_pairs {
                labels[k] = pair_label(alpha, beta, spin_pairs[(k, 0)], spin_pairs[(k, 1)]);
            }
            let mut phi = 0.0f64;
            for a in 0..n_pairs {
                let la = labels[a];
                if la == 0 {
                    continue;
                }
                for b in (a + 1)..n_pairs {
                    let lb = labels[b];
                    if lb == 0 {
                        continue;
                    }
                    if la == 1 && lb == 1 {
                        phi += theta_singlet[(a, b)];
                    } else if la == 2 && lb == 2 {
                        phi += theta_triplet[(a, b)];
                    }
                }
            }
            if phi != 0.0 {
                let (s, c) = phi.sin_cos();
                state[ia * dim_b + ib] *= Complex64::new(c, s);
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub fn apply_igcr4_pair_spin_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    theta_singlet: PyReadonlyArray2<f64>,
    theta_triplet: PyReadonlyArray2<f64>,
    norb: usize,
    alpha_occ: PyReadonlyArray2<u8>,
    beta_occ: PyReadonlyArray2<u8>,
    spin_pairs: PyReadonlyArray2<usize>,
) -> PyResult<()> {
    let mut vec_view = vec.as_array_mut();
    let dim_a = vec_view.shape()[0];
    let dim_b = vec_view.shape()[1];
    let alpha_occ = alpha_occ.as_array();
    let beta_occ = beta_occ.as_array();
    let spin_pairs = spin_pairs.as_array();
    let theta_singlet = theta_singlet.as_array();
    let theta_triplet = theta_triplet.as_array();

    if alpha_occ.shape()[0] != dim_a || beta_occ.shape()[0] != dim_b {
        return Err(PyValueError::new_err(
            "occupation dimensions must match state dimensions",
        ));
    }
    validate_spin_pairs(spin_pairs, norb)?;

    let alpha_bits = bits_from_occ(alpha_occ, norb)?;
    let beta_bits = bits_from_occ(beta_occ, norb)?;
    let mut state: Vec<Complex64> = vec_view.iter().copied().collect();

    apply_pair_basis_transform(&mut state, &alpha_bits, &beta_bits, dim_b, spin_pairs)?;
    apply_pair_spin_phases(
        &mut state,
        &alpha_bits,
        &beta_bits,
        dim_b,
        spin_pairs,
        theta_singlet,
        theta_triplet,
    )?;
    apply_pair_basis_transform(&mut state, &alpha_bits, &beta_bits, dim_b, spin_pairs)?;

    for (slot, value) in vec_view.iter_mut().zip(state.into_iter()) {
        *slot = value;
    }
    Ok(())
}
