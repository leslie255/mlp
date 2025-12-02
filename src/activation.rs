use std::fmt::{self, Debug};

#[derive(Clone, Copy)]
pub struct DynActivationFunction {
    name: &'static str,
    apply: fn(f32) -> f32,
    apply_multiple: unsafe fn(&[f32], &mut [f32]),
    deriv: fn(f32) -> f32,
}

impl Debug for DynActivationFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self.name, f)
    }
}

impl DynActivationFunction {
    pub fn new<Phi: ActivationFunction>(_: Phi) -> Self {
        Self {
            name: Phi::NAME,
            apply: Phi::apply,
            apply_multiple: Phi::apply_multiple,
            deriv: Phi::deriv,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn apply(&self, x: f32) -> f32 {
        (self.apply)(x)
    }

    /// # Safety
    ///
    /// `xs` and `ys` must be of the same length.
    pub unsafe fn apply_multiple(&self, xs: &[f32], ys: &mut [f32]) {
        unsafe { (self.apply_multiple)(xs, ys) }
    }

    pub fn deriv(&self, x: f32) -> f32 {
        (self.deriv)(x)
    }
}

pub trait ActivationFunction: Send + Sync + 'static {
    const NAME: &'static str;

    fn apply(x: f32) -> f32;

    fn deriv(x: f32) -> f32;

    fn apply_multiple(x: &[f32], y: &mut [f32]) {
        for i in 0..x.len() {
            y[i] = Self::apply(x[i]);
        }
    }
}

pub mod activation_functions {
    use super::ActivationFunction;

    use std::ptr::copy_nonoverlapping;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Identity;
    impl ActivationFunction for Identity {
        const NAME: &'static str = "identity";

        fn apply(x: f32) -> f32 {
            x
        }

        fn deriv(_: f32) -> f32 {
            1.0
        }

        fn apply_multiple(x: &[f32], y: &mut [f32]) {
            let len = x.len().min(y.len());
            unsafe {
                copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), len);
            }
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-x))
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Sigmoid;
    impl ActivationFunction for Sigmoid {
        const NAME: &'static str = "sigmoid";

        fn apply(x: f32) -> f32 {
            sigmoid(x)
        }

        fn deriv(x: f32) -> f32 {
            sigmoid(x) * (1.0 - sigmoid(x))
        }
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Tanh;
    impl ActivationFunction for Tanh {
        const NAME: &'static str = "tanh";

        fn apply(x: f32) -> f32 {
            f32::tanh(x)
        }

        fn deriv(x: f32) -> f32 {
            1.0 - f32::tanh(x).powi(2)
        }
    }
}
