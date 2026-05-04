use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::collections::HashSet;

fn popcount_u64(x: u64) -> usize {
    x.count_ones() as usize
}

fn alpha_mask(norb: usize) -> u64 {
    if norb == 64 {
        u64::MAX
    } else {
        (1u64 << norb) - 1
    }
}

fn split_spin_bits(bitstring: u64, norb: usize) -> (u64, u64) {
    let mask = alpha_mask(norb);
    let alpha = bitstring & mask;
    let beta = (bitstring >> norb) & mask;
    (alpha, beta)
}

fn join_spin_bits(alpha: u64, beta: u64, norb: usize) -> u64 {
    alpha | (beta << norb)
}

fn choose_weighted_index(weights: &[f64], rng: &mut Pcg64Mcg) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let mut r = rng.gen::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= *w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

fn repair_half(mut bits: u64, target_weight: usize, occ: &[f64], rng: &mut Pcg64Mcg) -> u64 {
    while popcount_u64(bits) > target_weight {
        let mut positions = Vec::new();
        let mut weights = Vec::new();
        for i in 0..occ.len() {
            if ((bits >> i) & 1) == 1 {
                positions.push(i);
                weights.push((1.0 - occ[i]).max(1e-12));
            }
        }
        if positions.is_empty() {
            break;
        }
        let k = choose_weighted_index(&weights, rng);
        bits &= !(1u64 << positions[k]);
    }

    while popcount_u64(bits) < target_weight {
        let mut positions = Vec::new();
        let mut weights = Vec::new();
        for i in 0..occ.len() {
            if ((bits >> i) & 1) == 0 {
                positions.push(i);
                weights.push(occ[i].max(1e-12));
            }
        }
        if positions.is_empty() {
            break;
        }
        let k = choose_weighted_index(&weights, rng);
        bits |= 1u64 << positions[k];
    }

    bits
}

#[pyfunction]
pub fn sample_indices_from_probabilities(
    probabilities: PyReadonlyArray1<f64>,
    n_samples: usize,
    seed: u64,
) -> Vec<u64> {
    let probs = probabilities.as_slice().unwrap();
    let mut cdf = Vec::with_capacity(probs.len());
    let mut acc = 0.0_f64;
    for &p in probs {
        acc += p.max(0.0);
        cdf.push(acc);
    }
    if acc <= 0.0 {
        return vec![0; n_samples];
    }
    for x in &mut cdf {
        *x /= acc;
    }

    let mut rng = Pcg64Mcg::new(seed as u128);
    let uni = Uniform::new(0.0_f64, 1.0_f64);
    let mut out = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let r = uni.sample(&mut rng);
        let idx = cdf.partition_point(|&x| x < r);
        out.push(idx as u64);
    }

    out
}

#[pyfunction]
pub fn postselect_spin_bitstrings(
    bitstrings: PyReadonlyArray1<u64>,
    norb: usize,
    n_alpha: usize,
    n_beta: usize,
) -> Vec<u64> {
    let xs = bitstrings.as_slice().unwrap();
    let mut out = Vec::new();
    for &x in xs {
        let (a, b) = split_spin_bits(x, norb);
        if popcount_u64(a) == n_alpha && popcount_u64(b) == n_beta {
            out.push(x);
        }
    }
    out
}

#[pyfunction]
pub fn estimate_spin_orbital_occupancies(
    bitstrings: PyReadonlyArray1<u64>,
    norb: usize,
) -> (Vec<f64>, Vec<f64>) {
    let xs = bitstrings.as_slice().unwrap();
    let n = xs.len().max(1) as f64;
    let mut occ_a = vec![0.0_f64; norb];
    let mut occ_b = vec![0.0_f64; norb];

    for &x in xs {
        let (a, b) = split_spin_bits(x, norb);
        for p in 0..norb {
            if ((a >> p) & 1) == 1 {
                occ_a[p] += 1.0;
            }
            if ((b >> p) & 1) == 1 {
                occ_b[p] += 1.0;
            }
        }
    }

    for p in 0..norb {
        occ_a[p] /= n;
        occ_b[p] /= n;
    }

    (occ_a, occ_b)
}

#[pyfunction]
pub fn recover_spin_bitstrings(
    bitstrings: PyReadonlyArray1<u64>,
    norb: usize,
    n_alpha: usize,
    n_beta: usize,
    occ_alpha: PyReadonlyArray1<f64>,
    occ_beta: PyReadonlyArray1<f64>,
    seed: u64,
) -> Vec<u64> {
    let xs = bitstrings.as_slice().unwrap();
    let occ_a = occ_alpha.as_slice().unwrap();
    let occ_b = occ_beta.as_slice().unwrap();

    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut out = Vec::with_capacity(xs.len());

    for &x in xs {
        let (a0, b0) = split_spin_bits(x, norb);
        let a = repair_half(a0, n_alpha, occ_a, &mut rng);
        let b = repair_half(b0, n_beta, occ_b, &mut rng);
        out.push(join_spin_bits(a, b, norb));
    }

    out
}

#[pyfunction]
pub fn subsample_batches(
    bitstrings: PyReadonlyArray1<u64>,
    batch_size: usize,
    num_batches: usize,
    seed: u64,
) -> Vec<Vec<u64>> {
    let xs = bitstrings.as_slice().unwrap();
    let mut uniq = Vec::new();
    let mut seen = HashSet::new();

    for &x in xs {
        if seen.insert(x) {
            uniq.push(x);
        }
    }

    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut out = Vec::with_capacity(num_batches);

    for _ in 0..num_batches {
        let mut batch = Vec::with_capacity(batch_size);
        if uniq.is_empty() {
            out.push(batch);
            continue;
        }
        for _ in 0..batch_size {
            let k = rng.gen_range(0..uniq.len());
            batch.push(uniq[k]);
        }
        out.push(batch);
    }

    out
}
