use mlp::{
    Gym, LayerDescription, NeuralNetwork, Topology, activation_functions::Sigmoid, faer::prelude::*,
};

fn main() {
    let training_samples: &[f32] = &[
        // An XOR gate.
        0., 0., 0., //
        0., 1., 1., //
        1., 0., 1., //
        1., 1., 0., //
    ];

    let mut nn = NeuralNetwork::new(Topology::new(
        2, // n_inputs
        [
            // layers
            LayerDescription::new(2, Sigmoid),
            LayerDescription::new(1, Sigmoid),
        ]
        .into(),
    ));

    nn.randomize_params(-0.1..0.1);

    let mut gym = Gym::new(&mut nn);

    for _ in 0..1_000_000 {
        // For this example, `train_singe_threaded` is actually faster than multi-threaded training
        // since the number of samples is quite low. Use `train` instead of `train_single_threaded`
        // for multi-threaded training.
        gym.train_single_threaded(
            // num_cpus::get(),  // n_threads
            0.25,             // eta
            training_samples, // samples
        );
    }

    println!("loss = {}", nn.loss(training_samples));

    println!("[Results]");
    for sample in training_samples.chunks(3) {
        let x = ColRef::from_slice(&sample[0..2]);
        let a = nn.forward(x);
        println!("{} xor {} = {}", x[0], x[1], a[0]);
    }
}
