#![feature(allocator_api)]

pub mod activation;
pub mod gym;
pub mod nn;
pub mod pretty_print;

use crate::{
    activation::*,
    nn::{LayerDescription, NeuralNetwork},
};

use faer::prelude::*;

fn main() {
    let training_data: &[(&[f32], &[f32])] = &[
        (&[0., 0.], &[0.]),
        (&[1., 0.], &[1.]),
        (&[0., 1.], &[1.]),
        (&[1., 1.], &[0.]),
    ];

    let mut nn = NeuralNetwork::new(
        2,
        [
            LayerDescription::new(24, Tanh),
            LayerDescription::new(24, Sigmoid),
            LayerDescription::new(32, Tanh),
            LayerDescription::new(32, Sigmoid),
            LayerDescription::new(1, Sigmoid),
        ],
    );

    nn.randomize(-0.2..0.2, -0.2..0.2);

    let mut gym = nn.go_to_gym();

    let eta = 0.8;
    let n_epochs = 1_000_000usize;
    for i_epoch in 0usize..n_epochs {
        let loss = gym.train(eta, training_data);
        if i_epoch % (n_epochs / n_epochs.min(20)) == 0 || i_epoch == n_epochs - 1 {
            let percentage = (i_epoch as f32) / (n_epochs as f32) * 100.0;
            println!("[{percentage:.0}%] L = {loss}");
        }
    }

    gym.finish();

    for i_layer in 1..=nn.n_layers() {
        let layer = nn.get_layer_mut(None, i_layer).unwrap();
        println!(
            "=== Layer #{i_layer} ===\n\n{}\n",
            layer.pretty_print_params(i_layer)
        );
    }

    for (i, (x_i, y_i)) in training_data.iter().enumerate() {
        nn.forward(ColRef::from_slice(x_i));
        let a_1 = nn.get_activation(nn.n_layers()).unwrap();
        println!("[i = {i}] x = {x_i:.0?}, y = {y_i:.0?}, a_1 = {a_1:.08?} ~ {a_1:.0?}");
    }
}
