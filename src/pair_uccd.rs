use ndarray::ArrayView1;
use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn apply_pair_uccd_doci_unitary_in_place(
    mut vec: PyReadwriteArray1<Complex64>,
    unitary: PyReadonlyArray2<Complex64>,
    indices: PyReadonlyArray1<usize>,
) -> PyResult<()> {
    let unitary = unitary.as_array();
    let indices = indices.as_array();
    let mut vec = vec.as_array_mut();

    if unitary.nrows() != unitary.ncols() {
        return Err(PyValueError::new_err("unitary must be square"));
    }
    if unitary.nrows() != indices.len() {
        return Err(PyValueError::new_err(
            "unitary dimension must match the number of DOCI indices",
        ));
    }

    let dim = indices.len();
    let mut subvec = vec![Complex64::new(0.0, 0.0); dim];
    for i in 0..dim {
        let idx = indices[i];
        if idx >= vec.len() {
            return Err(PyValueError::new_err("DOCI index out of bounds"));
        }
        subvec[i] = vec[idx];
    }

    let mut out = vec![Complex64::new(0.0, 0.0); dim];
    for i in 0..dim {
        let row: ArrayView1<'_, Complex64> = unitary.row(i);
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..dim {
            acc += row[j] * subvec[j];
        }
        out[i] = acc;
    }

    for i in 0..dim {
        vec[indices[i]] = out[i];
    }
    Ok(())
}
