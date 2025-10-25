pub trait ActivationFunction {
    fn apply(&self, x: f32) -> f32;
    fn deriv(&self, x: f32) -> f32;
    fn apply_vector<const N: usize>(&self, x: &[f32; N], y: &mut [f32; N]) {
        for i in 0..N {
            y[i] = self.apply(x[i]);
        }
    }
    fn deriv_vector<const N: usize>(&self, x: &[f32; N], y: &mut [f32; N]) {
        for i in 0..N {
            y[i] = self.apply(x[i]);
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Identity;
impl ActivationFunction for Identity {
    fn apply(&self, x: f32) -> f32 {
        x
    }

    fn deriv(&self, _: f32) -> f32 {
        1.0
    }
}

fn σ(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn apply(&self, x: f32) -> f32 {
        σ(x)
    }

    fn deriv(&self, x: f32) -> f32 {
        σ(x) * (1.0 - σ(x))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tanh;
impl ActivationFunction for Tanh {
    fn apply(&self, x: f32) -> f32 {
        f32::tanh(x)
    }

    fn deriv(&self, x: f32) -> f32 {
        1.0 - f32::tanh(x).powi(2)
    }
}
