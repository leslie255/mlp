use std::{mem::transmute, ptr::NonNull};

use faer::prelude::*;

/// A non-null pointer to a column vector.
pub struct ColPtr<T> {
    pub ptr: NonNull<T>,
    pub nrows: usize,
    pub row_stride: usize,
}

impl<T> Eq for ColPtr<T> {}

impl<T> PartialEq for ColPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.nrows == other.nrows
    }
}

impl<T> Clone for ColPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ColPtr<T> {}

impl<T> ColPtr<T> {
    /// # Safety
    ///
    /// - must meet safety requirements for performing `buffer.add(offset)`
    pub const unsafe fn with_offset(buffer: NonNull<T>, offset: usize, nrows: usize) -> Self {
        Self::new(unsafe { buffer.add(offset) }, nrows)
    }

    pub const fn new(ptr: NonNull<T>, nrows: usize) -> Self {
        Self {
            ptr,
            nrows,
            row_stride: 1,
        }
    }

    pub const fn from_col_ref(col_ref: ColRef<f32>) -> Self {
        unsafe { transmute(col_ref) }
    }

    pub const fn from_col_mut(col_mut: ColMut<f32>) -> Self {
        unsafe { transmute(col_mut) }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows` items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a`
    ///   reference
    pub const unsafe fn as_col_ref<'a>(self) -> ColRef<'a, T> {
        unsafe { transmute(self) }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows` items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a mut`
    ///   reference
    pub const unsafe fn as_col_mut<'a>(self) -> ColMut<'a, T> {
        unsafe { transmute(self) }
    }
}

/// A non-null pointer to a matrix.
pub struct MatPtr<T> {
    pub ptr: NonNull<T>,
    pub nrows: usize,
    pub ncols: usize,
    pub row_stride: usize,
    pub col_stride: isize,
}

impl<T> Eq for MatPtr<T> {}

impl<T> PartialEq for MatPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.nrows == other.nrows
    }
}

impl<T> Clone for MatPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for MatPtr<T> {}

impl<T> MatPtr<T> {
    /// # Safety
    ///
    /// - must meet safety requirements for performing `buffer.add(offset)`
    pub const unsafe fn with_offset(
        buffer: NonNull<T>,
        offset: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            ptr: unsafe { buffer.add(offset) },
            nrows,
            ncols,
            row_stride: 1,
            col_stride: nrows as isize,
        }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `f32` with at least `nrows * ncols`
    ///   items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a`
    ///   reference
    pub const unsafe fn as_mat_ref<'a>(self) -> MatRef<'a, f32> {
        unsafe { transmute(self) }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `f32` with at least `nrows * ncols`
    ///   items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a mut`
    ///   reference
    pub const unsafe fn as_mat_mut<'a>(self) -> MatMut<'a, f32> {
        unsafe { transmute(self) }
    }
}
