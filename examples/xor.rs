use std::{
    error::Error,
    fs,
    ptr::copy_nonoverlapping,
    time::{Duration, Instant},
};

use gnuplot::{AxesCommon, ColorType, Figure, PlotOption};
use mlp::{Gym, LayerDescription, NeuralNetwork, ParamBuffer, Typology, activation_functions::*};

use faer::prelude::*;

fn time<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let before = Instant::now();
    let result = f();
    let after = Instant::now();
    (after.duration_since(before), result)
}

struct LossRecords {
    i_epochs_records: Vec<f32>,
    loss_records: Vec<f32>,
}

fn plot_loss<'a>(records: &LossRecords, output_path: impl Into<Option<&'a str>>) {
    let output_path = output_path.into();
    match output_path {
        Some(output_path) => println!("Plotting loss progress to file {output_path:?}..."),
        None => println!("Plotting loss progress to gnuplot window..."),
    }
    let mut figure = Figure::new();
    figure.axes2d().set_y_log(Some(10.0)).lines(
        &records.i_epochs_records,
        &records.loss_records,
        &[
            PlotOption::Caption("Loss"),
            PlotOption::Color(ColorType::Black),
        ],
    );
    figure.set_title("Loss");
    if let Some(path) = output_path {
        figure.set_terminal("svg", path);
    }
    figure.show_and_keep_running().unwrap();
}

fn train(samples: &[(&[f32], &[f32])], nn: &mut NeuralNetwork, single_thread: bool) -> LossRecords {
    let eta = 0.2;

    let n_epochs = 1_000_000;
    let n_threads = match single_thread {
        true => 0,
        false => num_cpus::get(),
    };

    let n_logs = 20;
    let n_records = 1000;

    println!("Using {n_threads} threads");

    let mut i_epochs_records: Vec<f32> = Vec::with_capacity(n_records);
    let mut loss_records: Vec<f32> = Vec::with_capacity(n_records);

    let mut gym = Gym::new(nn);

    for i_epoch in 0usize..n_epochs {
        let loss = gym.train(n_threads, eta, samples);
        // Log.
        if i_epoch % (n_epochs / n_epochs.min(n_logs)) == 0 || i_epoch == n_epochs - 1 {
            let percentage = (i_epoch as f32) / (n_epochs as f32) * 100.0;
            println!("[{percentage:.0}%] L = {loss}");
        }
        // Record.
        if i_epoch % (n_epochs / n_epochs.min(n_records)) == 0 || i_epoch == n_epochs - 1 {
            i_epochs_records.push(i_epoch as f32);
            loss_records.push(loss);
        }
    }

    LossRecords {
        i_epochs_records,
        loss_records,
    }
}

fn load_params(param_buffer: &mut ParamBuffer) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read("./params.bin")?;
    let buffer: &[f32] = bytemuck::cast_slice(&bytes);
    let count = param_buffer.buffer_mut().len();
    unsafe {
        copy_nonoverlapping(
            buffer.as_ptr(),
            param_buffer.buffer_mut().as_mut_ptr(),
            count,
        );
    }
    Ok(())
}

fn dump_params(param_buffer: &ParamBuffer) -> Result<(), ()> {
    let buffer = param_buffer.buffer();
    let bytes: &[u8] = bytemuck::cast_slice(buffer);
    fs::write("./params.bin", bytes).map_err(|_| ())
}

fn main() {
    let samples: &[(&[f32], &[f32])] = &[
        (&[0., 0.], &[0.]),
        (&[1., 0.], &[1.]),
        (&[0., 1.], &[1.]),
        (&[1., 1.], &[0.]),
    ];

    let typology = Typology::new(
        2, // n_inputs
        [
            LayerDescription::new(128, Sigmoid),
            LayerDescription::new(128, Sigmoid),
            LayerDescription::new(1, Sigmoid),
        ]
        .into(),
    );

    let mut nn = NeuralNetwork::new(typology);

    // TODO: eliminate these unsafe with safe abstractions.
    match load_params(unsafe { nn.params_mut() }) {
        Ok(_) => {
            println!("Loaded parameters from `params.bin`");
        }
        Err(_) => {
            println!("Training from scratch");
            unsafe { nn.params_mut().randomize(-0.1..0.1) };
        }
    }

    println!("Training:");

    let (training_duration, records) = time(|| train(samples, &mut nn, true));
    println!("training took {training_duration:?}");

    dump_params(nn.params()).unwrap();

    plot_loss(&records, "loss_graph.svg");

    // Print parameters.
    // for i_layer in 0..nn.typology().n_layers() {
    //     println!(
    //         "=== Layer #{i_layer} ===\n\n{}\n",
    //         nn.params().pretty_print_layer(i_layer).unwrap(),
    //     );
    // }

    // Print results.
    for (i, (x_i, y_i)) in samples.iter().enumerate() {
        let x_i = ColRef::from_slice(x_i);
        let a_i = nn.forward(x_i);
        println!("[i = {i}] expected: {x_i:?} => {y_i:?}, result: {a_i:?}");
    }
}
