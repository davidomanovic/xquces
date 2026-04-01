use pyo3::prelude::*;

mod ucj_diag;
mod sqd;

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
        sqd::sample_indices_from_probabilities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sqd::postselect_spin_bitstrings,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sqd::estimate_spin_orbital_occupancies,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sqd::recover_spin_bitstrings,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sqd::subsample_batches,
        m
    )?)?;
    Ok(())
}