use std::collections::HashMap;

use ndarray::ArrayView2;
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy)]
struct PairHopTransition {
    source: usize,
    target: usize,
    pair_index: usize,
    sign: f64,
}

fn bits_from_occ(occ: ArrayView2<'_, u8>, norb: usize) -> PyResult<Vec<u64>> {
    if norb >= 63 {
        return Err(PyValueError::new_err(
            "gcr2 pair-hop Rust kernel supports norb < 63",
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

fn build_diag_values(
    alpha_bits: &[u64],
    beta_bits: &[u64],
    pair_params: &[f64],
    pairs: ArrayView2<'_, usize>,
) -> PyResult<Vec<f64>> {
    let dim_a = alpha_bits.len();
    let dim_b = beta_bits.len();
    let npairs = pairs.shape()[0];
    if pair_params.len() != npairs {
        return Err(PyValueError::new_err("pair_params has wrong length"));
    }
    if pairs.shape()[1] != 2 {
        return Err(PyValueError::new_err("pairs must have shape (n_pairs, 2)"));
    }

    let mut out = vec![0.0f64; dim_a * dim_b];
    for ia in 0..dim_a {
        let alpha = alpha_bits[ia];
        for ib in 0..dim_b {
            let beta = beta_bits[ib];
            let source = ia * dim_b + ib;
            let mut value = 0.0f64;
            for k in 0..npairs {
                let p = pairs[(k, 0)];
                let q = pairs[(k, 1)];
                let count_p = ((alpha >> p) & 1) as u32 + ((beta >> p) & 1) as u32;
                if count_p == 0 {
                    continue;
                }
                let count_q = ((alpha >> q) & 1) as u32 + ((beta >> q) & 1) as u32;
                if count_q != 0 {
                    value += pair_params[k] * (count_p * count_q) as f64;
                }
            }
            out[source] = value;
        }
    }
    Ok(out)
}

fn build_pair_hop_transitions(
    alpha_bits: &[u64],
    beta_bits: &[u64],
    pairs: ArrayView2<'_, usize>,
) -> PyResult<Vec<PairHopTransition>> {
    let dim_a = alpha_bits.len();
    let dim_b = beta_bits.len();
    let npairs = pairs.shape()[0];
    if pairs.shape()[1] != 2 {
        return Err(PyValueError::new_err("pairs must have shape (n_pairs, 2)"));
    }
    let map_a = build_index_map(alpha_bits);
    let map_b = build_index_map(beta_bits);
    let mut transitions = Vec::new();

    for k in 0..npairs {
        let p = pairs[(k, 0)];
        let q = pairs[(k, 1)];

        for ia in 0..dim_a {
            let alpha = alpha_bits[ia];
            let alpha_q_to_p = replace_orbital(alpha, q, p);
            let alpha_p_to_q = replace_orbital(alpha, p, q);

            for ib in 0..dim_b {
                let beta = beta_bits[ib];
                let source = ia * dim_b + ib;

                if let Some((new_alpha, sign_a)) = alpha_q_to_p {
                    if let Some((new_beta, sign_b)) = replace_orbital(beta, q, p) {
                        let target = map_a[&new_alpha] * dim_b + map_b[&new_beta];
                        transitions.push(PairHopTransition {
                            source,
                            target,
                            pair_index: k,
                            sign: sign_a * sign_b,
                        });
                    }
                }

                if let Some((new_alpha, sign_a)) = alpha_p_to_q {
                    if let Some((new_beta, sign_b)) = replace_orbital(beta, p, q) {
                        let target = map_a[&new_alpha] * dim_b + map_b[&new_beta];
                        transitions.push(PairHopTransition {
                            source,
                            target,
                            pair_index: k,
                            sign: -sign_a * sign_b,
                        });
                    }
                }
            }
        }
    }

    Ok(transitions)
}

fn apply_generator_scaled(
    input: &[Complex64],
    output: &mut [Complex64],
    diag_values: &[f64],
    pair_hop_params: &[f64],
    transitions: &[PairHopTransition],
    scale: f64,
) {
    for i in 0..output.len() {
        output[i] = input[i] * Complex64::new(0.0, scale * diag_values[i]);
    }
    for transition in transitions {
        let coeff = scale * pair_hop_params[transition.pair_index] * transition.sign;
        if coeff != 0.0 {
            output[transition.target] += input[transition.source] * coeff;
        }
    }
}

fn norm2(values: &[Complex64]) -> f64 {
    values.iter().map(|z| z.norm_sqr()).sum::<f64>()
}

fn generator_norm_bound(
    diag_values: &[f64],
    pair_hop_params: &[f64],
    transitions: &[PairHopTransition],
) -> f64 {
    let mut bound = diag_values
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));
    let hop_sum = pair_hop_params.iter().map(|x| x.abs()).sum::<f64>();
    if !transitions.is_empty() {
        bound += hop_sum;
    }
    bound
}

fn apply_exp_action_taylor(
    state: &mut [Complex64],
    diag_values: &[f64],
    pair_hop_params: &[f64],
    transitions: &[PairHopTransition],
    taylor_tol: f64,
    taylor_max_terms: usize,
) {
    let dim = state.len();
    let bound = generator_norm_bound(diag_values, pair_hop_params, transitions);
    let mut n_steps = 1usize;
    while bound / n_steps as f64 > 1.0 && n_steps < 8192 {
        n_steps *= 2;
    }
    let scale = 1.0 / n_steps as f64;
    let tol2 = taylor_tol * taylor_tol;

    let mut term = vec![Complex64::new(0.0, 0.0); dim];
    let mut next = vec![Complex64::new(0.0, 0.0); dim];
    let mut accum = vec![Complex64::new(0.0, 0.0); dim];

    for _ in 0..n_steps {
        term.copy_from_slice(state);
        accum.copy_from_slice(state);

        for k in 1..=taylor_max_terms {
            apply_generator_scaled(
                &term,
                &mut next,
                diag_values,
                pair_hop_params,
                transitions,
                scale,
            );
            let inv_k = 1.0 / k as f64;
            for i in 0..dim {
                term[i] = next[i] * inv_k;
                accum[i] += term[i];
            }
            if norm2(&term) <= tol2 * norm2(&accum).max(1.0e-300) {
                break;
            }
        }

        state.copy_from_slice(&accum);
    }
}

fn build_diag_values_from_features(
    diag_features: ArrayView2<'_, f64>,
    pair_params: &[f64],
) -> PyResult<Vec<f64>> {
    let dim = diag_features.shape()[0];
    let npairs = diag_features.shape()[1];
    if pair_params.len() != npairs {
        return Err(PyValueError::new_err(
            "pair_params length does not match diag_features",
        ));
    }
    let mut out = vec![0.0f64; dim];
    for i in 0..dim {
        let mut value = 0.0f64;
        for k in 0..npairs {
            value += diag_features[(i, k)] * pair_params[k];
        }
        out[i] = value;
    }
    Ok(out)
}

fn build_cached_transitions(
    transition_source: &[usize],
    transition_target: &[usize],
    transition_pair: &[usize],
    transition_sign: &[f64],
    dim: usize,
    npairs: usize,
) -> PyResult<Vec<PairHopTransition>> {
    let n = transition_source.len();
    if transition_target.len() != n || transition_pair.len() != n || transition_sign.len() != n {
        return Err(PyValueError::new_err(
            "transition arrays must have matching lengths",
        ));
    }
    let mut transitions = Vec::with_capacity(n);
    for i in 0..n {
        let source = transition_source[i];
        let target = transition_target[i];
        let pair_index = transition_pair[i];
        if source >= dim || target >= dim {
            return Err(PyValueError::new_err("transition index out of bounds"));
        }
        if pair_index >= npairs {
            return Err(PyValueError::new_err("transition pair index out of bounds"));
        }
        transitions.push(PairHopTransition {
            source,
            target,
            pair_index,
            sign: transition_sign[i],
        });
    }
    Ok(transitions)
}

#[pyfunction]
pub fn apply_gcr2_pairhop_middle_cached_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    pair_params: PyReadonlyArray1<f64>,
    pair_hop_params: PyReadonlyArray1<f64>,
    diag_features: PyReadonlyArray2<f64>,
    transition_source: PyReadonlyArray1<usize>,
    transition_target: PyReadonlyArray1<usize>,
    transition_pair: PyReadonlyArray1<usize>,
    transition_sign: PyReadonlyArray1<f64>,
    taylor_tol: f64,
    taylor_max_terms: usize,
) -> PyResult<()> {
    let mut vec_view = vec.as_array_mut();
    let dim = vec_view.shape()[0] * vec_view.shape()[1];
    let pair_params = pair_params.as_slice()?;
    let pair_hop_params = pair_hop_params.as_slice()?;
    let diag_features = diag_features.as_array();
    if diag_features.shape()[0] != dim {
        return Err(PyValueError::new_err(
            "diag_features first dimension must match flattened state size",
        ));
    }
    if pair_hop_params.len() != diag_features.shape()[1] {
        return Err(PyValueError::new_err(
            "pair_hop_params length does not match diag_features",
        ));
    }

    let diag_values = build_diag_values_from_features(diag_features, pair_params)?;
    let transitions = build_cached_transitions(
        transition_source.as_slice()?,
        transition_target.as_slice()?,
        transition_pair.as_slice()?,
        transition_sign.as_slice()?,
        dim,
        pair_hop_params.len(),
    )?;

    let mut state: Vec<Complex64> = vec_view.iter().copied().collect();
    apply_exp_action_taylor(
        &mut state,
        &diag_values,
        pair_hop_params,
        &transitions,
        taylor_tol,
        taylor_max_terms,
    );

    for (slot, value) in vec_view.iter_mut().zip(state.into_iter()) {
        *slot = value;
    }
    Ok(())
}

#[pyfunction]
pub fn apply_gcr2_pairhop_middle_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    pair_params: PyReadonlyArray1<f64>,
    pair_hop_params: PyReadonlyArray1<f64>,
    norb: usize,
    alpha_occ: PyReadonlyArray2<u8>,
    beta_occ: PyReadonlyArray2<u8>,
    pairs: PyReadonlyArray2<usize>,
    taylor_tol: f64,
    taylor_max_terms: usize,
) -> PyResult<()> {
    let mut vec_view = vec.as_array_mut();
    let dim_a = vec_view.shape()[0];
    let dim_b = vec_view.shape()[1];

    let alpha_occ = alpha_occ.as_array();
    let beta_occ = beta_occ.as_array();
    if alpha_occ.shape()[0] != dim_a || beta_occ.shape()[0] != dim_b {
        return Err(PyValueError::new_err(
            "occupation dimensions must match state dimensions",
        ));
    }

    let pair_params = pair_params.as_slice()?;
    let pair_hop_params = pair_hop_params.as_slice()?;
    let pairs = pairs.as_array();
    if pair_hop_params.len() != pairs.shape()[0] {
        return Err(PyValueError::new_err("pair_hop_params has wrong length"));
    }

    let alpha_bits = bits_from_occ(alpha_occ, norb)?;
    let beta_bits = bits_from_occ(beta_occ, norb)?;
    let diag_values = build_diag_values(&alpha_bits, &beta_bits, pair_params, pairs)?;
    let transitions = build_pair_hop_transitions(&alpha_bits, &beta_bits, pairs)?;

    let mut state: Vec<Complex64> = vec_view.iter().copied().collect();
    apply_exp_action_taylor(
        &mut state,
        &diag_values,
        pair_hop_params,
        &transitions,
        taylor_tol,
        taylor_max_terms,
    );

    for (slot, value) in vec_view.iter_mut().zip(state.into_iter()) {
        *slot = value;
    }
    Ok(())
}
