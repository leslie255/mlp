pub mod activation;
pub mod linear_algebra;

use std::ops::{Deref as _, DerefMut as _};

use activation::*;
use linear_algebra::*;
use rand::{Rng, distr::uniform::SampleRange};

pub trait UsizeList {}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UsizeListItem<const VALUE: usize, NextNode: UsizeList> {
    _marker: NextNode,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UsizeListEnd;

macro_rules! list {
    [] => {
        HyperListEnd
    };
    [$value:expr] => {
        HyperListItem<$value, HyperListEnd>
    };
    [$value:expr, $($xs:tt)*] => {
        HyperListItem<$value, list!($($xs)*)>
    };
}

#[derive(Debug, Clone, Default)]
pub struct NeuralNetwork<const N_0: usize, const N_1: usize, Phi: ActivationFunction = Identity> {
    pub phi: Phi,
    pub w: Box<Matrix<N_1, N_0>>,
    pub b: Box<Vector<N_1>>,
    pub z_1: Box<Vector<N_1>>,
    pub a_1: Box<Vector<N_1>>,
}

#[derive(Debug, Clone, Default)]
pub struct NeuralNetworkDeriv<const N_0: usize, const N_1: usize> {
    pub w: Box<Matrix<N_1, N_0>>,
    pub b: Box<Vector<N_1>>,
}

impl<const N_0: usize, const N_1: usize, Phi: ActivationFunction> NeuralNetwork<N_0, N_1, Phi> {
    pub fn randomize(
        &mut self,
        w_range: impl SampleRange<f32> + Clone,
        b_range: impl SampleRange<f32> + Clone,
    ) {
        let mut rng = rand::rng();
        for f in self.w.iter_elements_mut() {
            *f = rng.random_range(w_range.clone());
        }
        for f in self.b.iter_mut() {
            *f = rng.random_range(b_range.clone());
        }
    }

    pub fn forward(&mut self, x: &Vector<N_0>) {
        self.w.dot_vector(x, &mut self.z_1);
        self.z_1.add_in_place(&self.b);
        self.phi
            .apply_vector::<N_1>(self.z_1.deref().as_ref(), self.a_1.deref_mut().as_mut());
    }

    pub fn loss(&mut self, training_data: &[([f32; N_0], [f32; N_1])]) -> f32 {
        let n_t = training_data.len() as f32;
        training_data
            .iter()
            .map(|(x_i, y_i)| {
                let x_i = Vector::new_ref(x_i);
                let y_i = Vector::new_ref(y_i);
                self.forward(x_i);
                self.a_1
                    .iter()
                    .zip(y_i.iter())
                    .map(|(&a_k, &y_i_k)| (a_k - y_i_k).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / (n_t)
    }

    /// Trains for one epoch.
    /// Returns the loss.
    pub fn train(
        &mut self,
        deriv_buffer: &mut NeuralNetworkDeriv<N_0, N_1>,
        training_data: &[([f32; N_0], [f32; N_1])],
        eta: f32,
    ) -> f32 {
        let n_t = training_data.len() as f32;
        let mut loss = 0.0f32;
        deriv_buffer.w.clear();
        deriv_buffer.b.clear();
        for (x_i, y_i) in training_data {
            let x_i = Vector::new_ref(x_i);
            let y_i = Vector::new_ref(y_i);
            self.forward(x_i);
            for k in 0..N_1 {
                let a_k = self.a_1.get(k).unwrap();
                let z_k = self.z_1.get(k).unwrap();
                let y_i_k = y_i.get(k).unwrap();
                let b_k = deriv_buffer.b.get_mut(k).unwrap();
                let error = a_k - y_i_k;
                let phi_deriv = self.phi.deriv(z_k);
                loss += error.powi(2);
                *b_k += error * phi_deriv;
                for g in 0..N_0 {
                    let x_i_g = x_i.get(g).unwrap();
                    let w_k_g = deriv_buffer.w.get_element_mut(k, g).unwrap();
                    *w_k_g += error * phi_deriv * x_i_g;
                }
            }
        }
        loss /= n_t;
        deriv_buffer.b.mul_scalar_in_place(1.0 / n_t);
        deriv_buffer.w.mul_scalar_in_place(1.0 / n_t);
        self.b.add_in_place_scaled(-eta, &deriv_buffer.b);
        self.w.add_in_place_scaled(-eta, &deriv_buffer.w);
        loss
    }
}

fn main() {
    // let training_data: &[([f32; _], [f32; _])] = &[
    //     ([0., 0.], [1., 0., 0., 1.]),
    //     ([0., 1.], [1., 0., 0., 1.]),
    //     ([1., 0.], [1., 0., 0., 1.]),
    //     ([1., 1.], [0., 1., 1., 0.]),
    // ];

    let training_data: &[([f32; _], [f32; _])] = &[
        ([0., 0.], [0.]),
        ([0., 1.], [1.]),
        ([1., 0.], [1.]),
        ([1., 1.], [0.]),
    ];

    let mut nn = NeuralNetwork::<2, 1, Sigmoid>::default();
    let mut deriv_buffer = NeuralNetworkDeriv::<2, 1>::default();
    let n_epochs = 10_000_000;

    println!("# Training ({n_epochs} rounds):");
    for i_epoch in 0..n_epochs {
        let loss = nn.train(&mut deriv_buffer, training_data, 0.4);
        if i_epoch % (n_epochs / 20) == 0 || i_epoch == (n_epochs - 1) {
            let percent = i_epoch as f32 / n_epochs as f32 * 100.0;
            println!("[{percent:.0}%] L = {loss}");
        }
    }

    println!("# Result Network Parameters:");
    println!("W = [");
    for row in nn.w.iter_rows() {
        print!("    ");
        for element in row.iter() {
            print!("{element}, ");
        }
        println!();
    }
    println!("]");
    println!("b = {:?}", nn.b);

    println!("# Result:");
    for (i, (x_i, y_i)) in training_data.iter().enumerate() {
        let x_i = Vector::new_ref(x_i);
        let y_i = Vector::new_ref(y_i);
        nn.forward(x_i);
        let a = &nn.a_1;
        println!("[i = {i}] x = {x_i:?}, a(x) = {a:.04?}, y = {y_i:?}",)
    }
}
