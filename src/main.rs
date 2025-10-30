#![feature(allocator_api)]

pub mod activation;
pub mod nn;
pub mod pretty_print;

use crate::{
    activation::*,
    nn::{LayerDescription, NeuralNetwork},
};

use faer::prelude::*;

fn main() {
    let training_data: &[([f32; 2], [f32; 1])] = &[
        ([0., 0.], [0.]),
        ([1., 0.], [0.]),
        ([0., 1.], [0.]),
        ([1., 1.], [1.]),
    ];

    let layer_descriptions = [
        LayerDescription::new(2, Sigmoid),
        LayerDescription::new(1, Sigmoid),
    ];

    let mut nn = NeuralNetwork::new(2, layer_descriptions);

    nn.randomize(-0.2..0.2, -0.2..0.2);

    for i_layer in 1..=nn.n_layers() {
        let layer = nn.get_layer_mut(None, i_layer).unwrap();
        println!("=== Layer #{i_layer} ===\n\n{}\n", layer.pretty_print_params(i_layer));
    }

    for (i, (x_i, y_i)) in training_data.iter().enumerate() {
        nn.forward(ColRef::from_slice(x_i));
        let a_1 = nn.get_activation(1).unwrap();
        let a_2 = nn.get_activation(2).unwrap();
        println!("[i = {i}] x = {x_i:.0?}, y = {y_i:.0?}, a_1 = {a_1:.04?}, a_2 = {a_2:.04?}");
    }
}
