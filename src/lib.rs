use pyo3::prelude::*;

mod orbital_rotation;
mod gcr_pairhop;
mod igcr4_spin_combo6;
mod pair_uccd;
mod sqd;
mod ucj_diag;

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        ucj_diag::apply_ucj_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ucj_diag::apply_ucj_spin_balanced_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ucj_diag::apply_igcr2_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ucj_diag::apply_igcr3_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ucj_diag::apply_igcr4_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        igcr4_spin_combo6::apply_igcr4_spin_combo6_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gcr_pairhop::apply_gcr2_pairhop_middle_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gcr_pairhop::apply_gcr2_pairhop_middle_cached_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gcr_pairhop::apply_gcr2_pairhop_product_middle_cached_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pair_uccd::apply_pair_uccd_doci_unitary_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        orbital_rotation::apply_givens_rotation_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        orbital_rotation::apply_phase_shift_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sqd::sample_indices_from_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(sqd::postselect_spin_bitstrings, m)?)?;
    m.add_function(wrap_pyfunction!(sqd::estimate_spin_orbital_occupancies, m)?)?;
    m.add_function(wrap_pyfunction!(sqd::recover_spin_bitstrings, m)?)?;
    m.add_function(wrap_pyfunction!(sqd::subsample_batches, m)?)?;
    Ok(())
}
