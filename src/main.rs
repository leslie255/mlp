pub mod activation;

use activation::*;

#[derive(Debug, Clone, Default)]
pub struct NeuralNetwork<Φ: ActivationFunction = Identity> {
    pub φ: Φ,
    pub a_0: f32,
    pub w: f32,
    pub b: f32,
    pub z_1: f32,
    pub a_1: f32,
}

impl<Φ: ActivationFunction> NeuralNetwork<Φ> {
    pub fn forward(&mut self) {
        self.z_1 = self.a_0 * self.w + self.b;
        self.a_1 = self.φ.apply(self.z_1);
    }
}

/// The loss function.
/// In reality the loss by itself is quite useless, you probably want to use `deriv`, which
/// computes the partial derivations against loss, and loss itself, in one pass.
pub fn loss(nn: &mut NeuralNetwork<impl ActivationFunction>, training_data: &[(f32, f32)]) -> f32 {
    let n = training_data.len();
    let mut loss = 0.0f32;
    for &(x_i, y_i) in training_data {
        nn.a_0 = x_i;
        nn.forward();
        let l_i = (nn.a_1 - y_i).powi(2);
        loss += l_i;
    }
    loss /= n as f32;
    loss
}

/// Returns the loss.
pub fn deriv<Φ: ActivationFunction>(
    nn: &mut NeuralNetwork<Φ>,
    nn_deriv: &mut NeuralNetwork<impl ActivationFunction>,
    training_data: &[(f32, f32)],
) -> f32 {
    nn_deriv.w = 0.0;
    nn_deriv.b = 0.0;
    let n = training_data.len() as f32;
    let mut loss = 0.0f32;
    for &(x_i, y_i) in training_data {
        nn.a_0 = x_i;
        nn.forward();
        let error = nn.a_1 - y_i;
        loss += error.powi(2);
        let φ_deriv = nn.φ.deriv(nn.z_1);
        nn_deriv.b += error * φ_deriv;
        nn_deriv.w += error * φ_deriv * x_i;
    }
    loss /= n;
    nn_deriv.w /= n;
    nn_deriv.b /= n;
    loss
}

pub fn train<Φ: ActivationFunction>(
    nn: &mut NeuralNetwork<Φ>,
    training_data: &[(f32, f32)],
    η: f32,
    n_iterations: usize,
) {
    // Activation function doesn't matter here.
    let mut nn_deriv = NeuralNetwork::<Identity>::default();
    for _ in 0..n_iterations {
        deriv(nn, &mut nn_deriv, training_data);
        nn.w -= nn_deriv.w * η;
        nn.b -= nn_deriv.b * η;
    }
}

fn main() {
    // Train the network to be a NOT gate.
    let training_data: &[(f32, f32)] = &[
        (0.0, 1.0), //
        (1.0, 0.0), //
    ];
    let mut nn = NeuralNetwork::<Sigmoid>::default();
    let η = 0.1;
    let n_epochs = 10_000_000;
    train(&mut nn, training_data, η, n_epochs);

    // Print results.
    for (i, &(x_i, y_i)) in training_data.iter().enumerate() {
        nn.a_0 = x_i;
        nn.forward();
        println!(
            "[i = {i:.08}] x_i = {x_i:.08}, a_1 = {:.08}, y_i = {y_i:.08}",
            nn.a_1
        );
    }
    let loss = loss(&mut nn, training_data);
    println!("L = {loss:.08}");
}
