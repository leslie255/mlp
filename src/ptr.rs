use std::ptr::NonNull;

use faer::prelude::*;

/// A non-null pointer to a column vector.
pub struct ColPtr<T> {
    pub ptr: NonNull<T>,
    pub nrows: usize,
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
    /// - this slice of `T` must satisfy aliasing requirements for being cast into a `&'a`
    ///   reference
    pub const unsafe fn as_col_ref<'a>(self) -> ColRef<'a, T> {
        unsafe { ColRef::from_raw_parts(self.ptr.as_ptr(), self.nrows, 1) }
    }

    /// # Safety
    ///
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows` items
    /// - this slice of `T` must satisfy aliasing requirements for being cast into a `&'a mut`
    ///   reference
    pub const unsafe fn as_col_mut<'a>(self) -> ColMut<'a, T> {
        unsafe { ColMut::from_raw_parts_mut(self.ptr.as_ptr(), self.nrows, 1) }
    }
}

/// A non-null pointer to a matrix.
pub struct MatPtr<T> {
    pub ptr: NonNull<T>,
    pub nrows: usize,
    pub ncols: usize,
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
    /// - must meet safety requirements for performing `data.add(offset)`
    pub const unsafe fn with_offset(
        data: NonNull<T>,
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
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows * ncols`
    ///   items
    /// - this slice of `T` must satisfy aliasing requirements for being cast into a `&'a`
    ///   reference
    pub const unsafe fn as_mat_ref<'a>(self) -> MatRef<'a, T> {
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
    /// - `ptr` must be pointing to a beginning of a slice of `T` with at least `nrows * ncols`
    ///   items
    /// - this slice of `T` must satisfy aliasing requirements for being cast into a `&'a mut`
    ///   reference
    pub const unsafe fn as_mat_mut<'a>(self) -> MatMut<'a, T> {
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
