use pyo3::prelude::*;

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
    Ok(())
}