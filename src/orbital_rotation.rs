use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn apply_phase_shift_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    phase_shift_re: f64,
    phase_shift_im: f64,
    indices: PyReadonlyArray1<usize>,
) -> PyResult<()> {
    let mut vec = vec.as_array_mut();
    let ncols = vec.shape()[1];
    let data = vec
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("vec must be C-contiguous"))?;
    let ptr_addr = data.as_mut_ptr() as usize;
    let rows: Vec<usize> = indices.as_array().iter().copied().collect();
    let phase_shift = Complex64::new(phase_shift_re, phase_shift_im);

    rows.par_iter().for_each(|&row| unsafe {
        let ptr = ptr_addr as *mut Complex64;
        let row_ptr = ptr.add(row * ncols);
        for col in 0..ncols {
            *row_ptr.add(col) *= phase_shift;
        }
    });

    Ok(())
}

#[pyfunction]
pub fn apply_givens_rotation_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    c: f64,
    s_re: f64,
    s_im: f64,
    slice1: PyReadonlyArray1<usize>,
    slice2: PyReadonlyArray1<usize>,
) -> PyResult<()> {
    let slice1 = slice1.as_array();
    let slice2 = slice2.as_array();

    if slice1.len() != slice2.len() {
        return Err(PyValueError::new_err("slice1 and slice2 must have the same length"));
    }

    let mut vec = vec.as_array_mut();
    let ncols = vec.shape()[1];
    let data = vec
        .as_slice_mut()
        .ok_or_else(|| PyValueError::new_err("vec must be C-contiguous"))?;
    let ptr_addr = data.as_mut_ptr() as usize;
    let s = Complex64::new(s_re, s_im);

    let pairs: Vec<(usize, usize)> = slice1
        .iter()
        .copied()
        .zip(slice2.iter().copied())
        .collect();

    pairs.par_iter().for_each(|&(i, j)| unsafe {
        let ptr = ptr_addr as *mut Complex64;
        let row_i = ptr.add(i * ncols);
        let row_j = ptr.add(j * ncols);
        for col in 0..ncols {
            let x = *row_i.add(col);
            let y = *row_j.add(col);
            *row_i.add(col) = x * c + y * s;
            *row_j.add(col) = y * c - x * s.conj();
        }
    });

    Ok(())
}
