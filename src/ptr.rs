use std::ptr::NonNull;

use faer::prelude::*;

/// A non-null pointer to a column vector.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ColPtr<T> {
    pub ptr: NonNull<T>,
    pub nrows: usize,
}

impl<T> ColPtr<T> {
    /// # Safety
    ///
    /// - must meet safety requirements for performing `data.add(offset)`
    pub const unsafe fn with_offset(data: NonNull<T>, offset: usize, nrows: usize) -> Self {
        Self {
            ptr: unsafe { data.add(offset) },
            nrows,
        }
    }

    pub const fn new(ptr: NonNull<T>, nrows: usize) -> Self {
        Self { ptr, nrows }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows` items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a`
    ///   reference
    pub const unsafe fn as_col_ref<'a>(self) -> ColRef<'a, T> {
        unsafe { ColRef::from_raw_parts(self.ptr.as_ptr(), self.nrows, 1) }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows` items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a mut`
    ///   reference
    pub const unsafe fn as_col_mut<'a>(self) -> ColMut<'a, T> {
        unsafe { ColMut::from_raw_parts_mut(self.ptr.as_ptr(), self.nrows, 1) }
    }
}

/// A non-null pointer to a matrix.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MatPtr {
    pub ptr: NonNull<f32>,
    pub nrows: usize,
    pub ncols: usize,
}

impl MatPtr {
    /// # Safety
    ///
    /// - must meet safety requirements for performing `data.add(offset)`
    pub const unsafe fn with_offset(
        data: NonNull<f32>,
        offset: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            ptr: unsafe { data.add(offset) },
            nrows,
            ncols,
        }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `f32` with at least `nrows * ncols`
    ///   items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a`
    ///   reference
    pub const unsafe fn as_mat_ref<'a>(self) -> MatRef<'a, f32> {
        unsafe {
            MatRef::from_raw_parts(
                self.ptr.as_ptr().cast(),
                self.nrows,
                self.ncols,
                1,
                self.nrows as isize,
            )
        }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `f32` with at least `nrows * ncols`
    ///   items
    /// - this slice of `f32` must satisfy aliasing requirements for being cast into a `&'a mut`
    ///   reference
    pub const unsafe fn as_mat_mut<'a>(self) -> MatMut<'a, f32> {
        unsafe {
            MatMut::from_raw_parts_mut(
                self.ptr.as_ptr().cast(),
                self.nrows,
                self.ncols,
                1,
                self.nrows as isize,
            )
        }
    }
}


