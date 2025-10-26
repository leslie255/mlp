use std::{
    alloc::{Layout, alloc_zeroed},
    fmt::{self, Debug},
};

use crate::big_matrices::Vector;

/// A large matrix.
/// M: Number of rows / height of matrix.
/// N: Number of columns / width of matrix / size of each row vector.
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Matrix<const M: usize, const N: usize> {
    pub rows: [Vector<N>; M],
}

impl<const M: usize, const N: usize> Debug for Matrix<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter_rows()).finish()
    }
}

impl<const M: usize, const N: usize> Default for Box<Matrix<M, N>> {
    fn default() -> Self {
        Matrix::new_boxed_zeroed()
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    /// Initializes a matrix of size M×N, all zeroes.
    pub fn new_boxed_zeroed() -> Box<Self> {
        let bytes = unsafe { alloc_zeroed(Layout::new::<Self>()) };
        // Safety: `f32` is zero-initializable, and `Self` is repr(transparent).
        unsafe { Box::from_raw(bytes as *mut Self) }
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self, 0u8, size_of::<Self>());
        }
    }

    pub fn get_row(&self, i_m: usize) -> Option<&Vector<N>> {
        self.rows.get(i_m)
    }

    pub fn get_row_mut(&mut self, i_m: usize) -> Option<&mut Vector<N>> {
        self.rows.get_mut(i_m)
    }

    /// # Safety
    ///
    /// `i_m` must be in range.
    pub unsafe fn get_row_unchecked(&self, i_m: usize) -> &Vector<N> {
        unsafe { self.rows.get_unchecked(i_m) }
    }

    /// # Safety
    ///
    /// `i_m` must be in range.
    pub unsafe fn get_row_unchecked_mut(&mut self, i_m: usize) -> &mut Vector<N> {
        unsafe { self.rows.get_unchecked_mut(i_m) }
    }

    pub fn get_element(&self, i_m: usize, i_n: usize) -> Option<f32> {
        self.get_row(i_m).and_then(|row| row.get(i_n))
    }

    pub fn get_element_mut(&mut self, i_m: usize, i_n: usize) -> Option<&mut f32> {
        self.get_row_mut(i_m).and_then(|row| row.get_mut(i_n))
    }

    /// # Safety
    ///
    /// `i_m`, `i_n` must be in range.
    pub unsafe fn get_element_unchecked(&self, i_m: usize, i_n: usize) -> f32 {
        unsafe { self.get_row_unchecked(i_m).get_unchecked(i_n) }
    }

    /// # Safety
    ///
    /// `i_m`, `i_n` must be in range.
    pub unsafe fn get_element_unchecked_mut(&mut self, i_m: usize, i_n: usize) -> &mut f32 {
        unsafe { self.get_row_unchecked_mut(i_m).get_unchecked_mut(i_n) }
    }

    pub fn iter_rows<'a>(&'a self) -> impl Iterator<Item = &'a Vector<N>> + use<'a, M, N> {
        self.rows.iter()
    }

    pub fn iter_rows_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut Vector<N>> + use<'a, M, N> {
        self.rows.iter_mut()
    }

    pub fn iter_elements<'a>(&'a self) -> impl Iterator<Item = &'a f32> + use<'a, M, N> {
        self.rows.iter().flat_map(|row| row.iter())
    }

    pub fn iter_elements_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut f32> + use<'a, M, N> {
        self.rows.iter_mut().flat_map(|row| row.iter_mut())
    }

    pub fn add(&self, other: &Self, result: &mut Self) {
        self.binary_op(other, result, |lhs, rhs| lhs + rhs);
    }

    pub fn add_in_place(&mut self, other: &Self) {
        self.binary_op_in_place(other, |lhs, rhs| lhs + rhs);
    }

    pub fn sub(&self, other: &Self, result: &mut Self) {
        self.binary_op(other, result, |lhs, rhs| lhs - rhs);
    }

    pub fn sub_in_place(&mut self, other: &Self) {
        self.binary_op_in_place(other, |lhs, rhs| lhs - rhs);
    }

    pub fn add_in_place_scaled(&mut self, scalar: f32, other: &Self) {
        self.binary_op_in_place(other, |lhs, rhs| lhs + rhs * scalar);
    }

    pub fn mul_scalar(&self, scalar: f32, result: &mut Self) {
        self.unary_op(result, |x| x * scalar);
    }

    pub fn mul_scalar_in_place(&mut self, scalar: f32) {
        self.unary_op_in_place(|x| x * scalar);
    }

    pub fn add_scalar(&self, scalar: f32, result: &mut Self) {
        self.unary_op(result, |x| x + scalar);
    }

    pub fn add_scalar_in_place(&mut self, scalar: f32) {
        self.unary_op_in_place(|x| x + scalar);
    }

    pub fn neg(&self, result: &mut Self) {
        self.unary_op(result, |x| -x);
    }

    pub fn neg_in_place(&mut self) {
        self.unary_op_in_place(|x| -x);
    }

    #[inline(always)]
    fn unary_op(&self, result: &mut Self, mut f: impl FnMut(f32) -> f32) {
        for i_m in 0..M {
            let self_row = &self.rows[i_m];
            let result_row = &mut result.rows[i_m];
            for i_n in 0..N {
                let res = &mut result_row.elements[i_n];
                let x = self_row.elements[i_n];
                *res = f(x);
            }
        }
    }

    #[inline(always)]
    fn unary_op_in_place(&mut self, mut f: impl FnMut(f32) -> f32) {
        for i_m in 0..M {
            let row = &mut self.rows[i_m];
            for i_n in 0..N {
                let x = &mut row.elements[i_n];
                *x = f(*x);
            }
        }
    }

    #[inline(always)]
    fn binary_op(&self, other: &Self, result: &mut Self, mut f: impl FnMut(f32, f32) -> f32) {
        for i_m in 0..M {
            let self_row = &self.rows[i_m];
            let other_row = &other.rows[i_m];
            let result_row = &mut result.rows[i_m];
            for i_n in 0..N {
                let res = &mut result_row.elements[i_n];
                let lhs = self_row.elements[i_n];
                let rhs = other_row.elements[i_n];
                *res = f(lhs, rhs);
            }
        }
    }

    #[inline(always)]
    fn binary_op_in_place(&mut self, other: &Self, mut f: impl FnMut(f32, f32) -> f32) {
        for i_m in 0..M {
            let self_row = &mut self.rows[i_m];
            let other_row = &other.rows[i_m];
            for i_n in 0..N {
                let lhs = &mut self_row.elements[i_n];
                let rhs = other_row.elements[i_n];
                *lhs = f(*lhs, rhs);
            }
        }
    }

    pub fn dot_vector(&self, vector: &Vector<N>, result: &mut Vector<M>) {
        for i_m in 0..M {
            let v_lhs = &self.rows[i_m];
            let v_rhs = vector;
            let s_result = v_lhs.dot_vector(v_rhs);
            result.elements[i_m] = s_result;
        }
    }
}
