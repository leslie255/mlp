use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{BufWriter, Write as _}, time::{Duration, Instant},
};

use mlp::{LayerDescription, NeuralNetwork, activation_functions::Sigmoid};

use faer::prelude::*;

fn time(f: impl FnOnce()) -> Duration {
    let before = Instant::now();
    f();
    let after = Instant::now();
    after.duration_since(before)
}

fn train(training_data: &[(&[f32], &[f32])], nn: &mut NeuralNetwork, log: bool, record: bool) {
    let mut gym = nn.go_to_gym();

    let eta = 0.2;
    let n_epochs = 1_000_000usize;
    let n_records = 1000usize;
    let mut loss_records: Vec<(usize, f32)> =
        Vec::with_capacity(if record { n_records } else { 0 });
    for i_epoch in 0usize..n_epochs {
        let loss = if i_epoch == 0 {
            gym.train(eta, training_data)
        } else {
            unsafe { gym.train_unchecked(eta, training_data) }
        };
        if record
            && (i_epoch % (n_epochs / n_epochs.min(n_records)) == 0 || i_epoch == n_epochs - 1)
        {
            loss_records.push((i_epoch, loss));
        }
        if log && (i_epoch % (n_epochs / n_epochs.min(20)) == 0 || i_epoch == n_epochs - 1) {
            let percentage = (i_epoch as f32) / (n_epochs as f32) * 100.0;
            println!("[{percentage:.0}%] L = {loss}");
        }
    }

    if record {
        let mut file = BufWriter::new(
            OpenOptions::new()
                .read(false)
                .write(true)
                .create(true)
                .truncate(true)
                .open("./loss_records.txt")
                .unwrap(),
        );
        writeln!(&mut file, "i_epoch\tloss").unwrap();
        for (i_epoch, loss) in loss_records {
            writeln!(&mut file, "{i_epoch}\t{loss}").unwrap();
        }
    }

    gym.finish();
}

fn load_params(nn: &mut NeuralNetwork) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read("./params.bin")?;
    let buffer: &[f32] = bytemuck::cast_slice(&bytes);
    nn.load_params(buffer)?;
    Ok(())
}

fn dump_params(nn: &NeuralNetwork) -> Result<(), ()> {
    let buffer = nn.params_buffer();
    let bytes: &[u8] = bytemuck::cast_slice(buffer);
    fs::write("./params.bin", bytes).map_err(|_| ())
}

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
            LayerDescription::new(12, Sigmoid),
            LayerDescription::new(24, Sigmoid),
            LayerDescription::new(24, Sigmoid),
            LayerDescription::new(1, Sigmoid),
        ],
    );

    match load_params(&mut nn) {
        Ok(_) => {
            println!("Loaded parameters from `params.bin`");
        }
        Err(_) => {
            println!("Training from scratch");
            nn.randomize(-0.2..0.2, -0.2..0.2);
        }
    }

    let training_duration = time(|| train(training_data, &mut nn, true, false));
    println!("training took {training_duration:?}");

    dump_params(&nn).unwrap();

    // for i_layer in 1..=nn.n_layers() {
    //     let layer = nn.layer_mut(i_layer).unwrap();
    //     println!(
    //         "=== Layer #{i_layer} ===\n\n{}\n",
    //         layer.pretty_print_params(i_layer)
    //     );
    // }

    for (i, (x_i, y_i)) in training_data.iter().enumerate() {
        let a = nn.forward(ColRef::from_slice(x_i));
        println!("[i = {i}] x = {x_i:?}, y = {y_i:?}, a = {a:.08?} ~ {a:.0?}");
    }
}
