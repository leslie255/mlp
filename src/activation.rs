use std::{
    any::type_name,
    fmt::{self, Debug},
};

pub trait ActivationFunction<T>: Send + Sync + 'static {
    fn name() -> &'static str {
        type_name::<Self>()
    }

    fn apply(x: T) -> T;

    fn deriv(x: T) -> T;
}

pub struct ActivationFunctionVTable<T> {
    pub name: fn() -> &'static str,
    pub apply: fn(T) -> T,
    pub deriv: fn(T) -> T,
}

impl<T> Debug for ActivationFunctionVTable<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&(self.name)(), f)
    }
}

impl<T> Clone for ActivationFunctionVTable<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ActivationFunctionVTable<T> {}

impl<T> ActivationFunctionVTable<T> {
    pub(crate) fn new<Phi: ActivationFunction<T>>(_: Phi) -> Self {
        Self {
            name: Phi::name,
            apply: Phi::apply,
            deriv: Phi::deriv,
        }
    }
}

pub mod activation_functions {
    use super::ActivationFunction;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Identity;
    impl ActivationFunction<f32> for Identity {
        fn name() -> &'static str {
            "identity"
        }

        fn apply(x: f32) -> f32 {
            x
        }

        fn deriv(_: f32) -> f32 {
            1.0
        }
    }

    fn sigmoid_f128(x: f128) -> f128 {
        1.0 / (1.0 + f128::exp(-x))
    }

    fn sigmoid_f64(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }

    fn sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-x))
    }

    fn sigmoid_f16(x: f16) -> f16 {
        1.0 / (1.0 + f16::exp(-x))
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Sigmoid;
    impl ActivationFunction<f128> for Sigmoid {
        fn name() -> &'static str {
            "sigmoid"
        }
        fn apply(x: f128) -> f128 {
            sigmoid_f128(x)
        }
        fn deriv(x: f128) -> f128 {
            sigmoid_f128(x) * (1.0 - sigmoid_f128(x))
        }
    }
    impl ActivationFunction<f64> for Sigmoid {
        fn name() -> &'static str {
            "sigmoid"
        }
        fn apply(x: f64) -> f64 {
            sigmoid_f64(x)
        }
        fn deriv(x: f64) -> f64 {
            sigmoid_f64(x) * (1.0 - sigmoid_f64(x))
        }
    }
    impl ActivationFunction<f32> for Sigmoid {
        fn name() -> &'static str {
            "sigmoid"
        }
        fn apply(x: f32) -> f32 {
            sigmoid_f32(x)
        }
        fn deriv(x: f32) -> f32 {
            sigmoid_f32(x) * (1.0 - sigmoid_f32(x))
        }
    }
    impl ActivationFunction<f16> for Sigmoid {
        fn name() -> &'static str {
            "sigmoid"
        }
        fn apply(x: f16) -> f16 {
            sigmoid_f16(x)
        }
        fn deriv(x: f16) -> f16 {
            sigmoid_f16(x) * (1.0 - sigmoid_f16(x))
        }
    }

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct Tanh;
    impl ActivationFunction<f32> for Tanh {
        fn name() -> &'static str {
            "tanh"
        }

        fn apply(x: f32) -> f32 {
            f32::tanh(x)
        }

        fn deriv(x: f32) -> f32 {
            1.0 - f32::tanh(x).powi(2)
        }
    }
}
