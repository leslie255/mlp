use std::{
    alloc::{alloc_zeroed, Layout}, fmt::{self, Debug}, mem::transmute
};

/// A large vector.
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vector<const N: usize> {
    pub elements: [f32; N],
}

impl<const N: usize> Debug for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.elements, f)
    }
}

impl<const N: usize> Default for Box<Vector<N>> {
    fn default() -> Self {
        Vector::new_boxed_zeroed()
    }
}

impl<const N: usize> AsRef<[f32; N]> for Vector<N> {
    fn as_ref(&self) -> &[f32; N] {
        &self.elements
    }
}

impl<const N: usize> AsMut<[f32; N]> for Vector<N> {
    fn as_mut(&mut self) -> &mut [f32; N] {
        &mut self.elements
    }
}

impl<const N: usize> Vector<N> {
    /// Initializes a vector of size N, all zeroes.
    pub fn new_boxed_zeroed() -> Box<Self> {
        let bytes = unsafe { alloc_zeroed(Layout::new::<Self>()) };
        // Safety: `f32` is zero-initializable, and `Self` is repr(C).
        unsafe { Box::from_raw(bytes as *mut Self) }
    }

    pub fn new_ref(array: &[f32; N]) -> &Self {
        unsafe { transmute(array) }
    }

    pub fn new_mut(array: &mut [f32; N]) -> &mut Self {
        unsafe { transmute(array) }
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self, 0u8, size_of::<Self>());
        }
    }

    pub fn get(&self, i: usize) -> Option<f32> {
        self.elements.get(i).copied()
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut f32> {
        self.elements.get_mut(i)
    }

    /// # Safety
    ///
    /// `i` must be in range.
    pub unsafe fn get_unchecked(&self, i: usize) -> f32 {
        unsafe { *self.elements.get_unchecked(i) }
    }

    /// # Safety
    ///
    /// `i` must be in range.
    pub unsafe fn get_unchecked_mut(&mut self, i: usize) -> &mut f32 {
        unsafe { self.elements.get_unchecked_mut(i) }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a f32> + use<'a, N> {
        self.elements.iter()
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut f32> + use<'a, N> {
        self.elements.iter_mut()
    }

    pub fn add(&self, other: &Vector<N>, result: &mut Self) {
        self.binary_op(other, result, |lhs, rhs| lhs + rhs);
    }

    pub fn add_in_place(&mut self, other: &Self) {
        self.binary_op_in_place(other, |lhs, rhs| lhs + rhs);
    }

    pub fn sub(&self, other: &Vector<N>, result: &mut Self) {
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
        for i in 0..N {
            let result = &mut result.elements[i];
            let x = self.elements[i];
            *result = f(x);
        }
    }

    #[inline(always)]
    fn unary_op_in_place(&mut self, mut f: impl FnMut(f32) -> f32) {
        for i in 0..N {
            let x = &mut self.elements[i];
            *x = f(*x);
        }
    }

    #[inline(always)]
    fn binary_op(&self, other: &Self, result: &mut Self, mut f: impl FnMut(f32, f32) -> f32) {
        for i in 0..N {
            let result = &mut result.elements[i];
            let lhs = self.elements[i];
            let rhs = other.elements[i];
            *result = f(lhs, rhs);
        }
    }

    #[inline(always)]
    fn binary_op_in_place(&mut self, other: &Self, mut f: impl FnMut(f32, f32) -> f32) {
        for i in 0..N {
            let lhs = &mut self.elements[i];
            let rhs = other.elements[i];
            let result = f(*lhs, rhs);
            *lhs = result;
        }
    }

    pub fn dot_vector(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..N {
            let lhs = self.elements[i];
            let rhs = other.elements[i];
            sum += lhs * rhs;
        }
        sum
    }
}
