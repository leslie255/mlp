use std::any::type_name_of_val;

pub trait ActivationFunction: Send + Sync + 'static {
    fn name(&self) -> &'static str {
        type_name_of_val(self)
    }

    fn apply(&self, x: f32) -> f32;

    fn deriv(&self, x: f32) -> f32;

    fn apply_vector(&self, x: &[f32], y: &mut [f32]) {
        for i in 0..x.len() {
            y[i] = self.apply(x[i]);
        }
    }
}

pub mod activation_functions {
    use super::ActivationFunction;

    use std::ptr::copy_nonoverlapping;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Identity;
    impl ActivationFunction for Identity {
        fn name(&self) -> &'static str {
            "identity"
        }

        fn apply(&self, x: f32) -> f32 {
            x
        }

        fn deriv(&self, _: f32) -> f32 {
            1.0
        }

        fn apply_vector(&self, x: &[f32], y: &mut [f32]) {
            let len = x.len().min(y.len());
            unsafe {
                copy_nonoverlapping(x.as_ptr(), y.as_mut_ptr(), len);
            }
        }
    }

    fn sigma(x: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-x))
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Sigmoid;
    impl ActivationFunction for Sigmoid {
        fn name(&self) -> &'static str {
            "sigmoid"
        }

        fn apply(&self, x: f32) -> f32 {
            sigma(x)
        }

        fn deriv(&self, x: f32) -> f32 {
            sigma(x) * (1.0 - sigma(x))
        }
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Tanh;
    impl ActivationFunction for Tanh {
        fn name(&self) -> &'static str {
            "tanh"
        }

        fn apply(&self, x: f32) -> f32 {
            f32::tanh(x)
        }

        fn deriv(&self, x: f32) -> f32 {
            1.0 - f32::tanh(x).powi(2)
        }
    }
}
