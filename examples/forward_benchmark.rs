#![feature(f16, f128)]

use std::{io::{Write, stdout}, time::Instant};

use mlp::{LayerDescription, NeuralNetwork, activation_functions::Sigmoid};

fn main() {
    let n_times = 10_000_000u32;

    macro_rules! benchmark {
        ($T:ty) => {{
            let type_name = stringify!($T);
            let mut nn = NeuralNetwork::<$T>::new(
                4,
                [
                    LayerDescription::new(8, Sigmoid),
                    LayerDescription::new(8, Sigmoid),
                    LayerDescription::new(1, Sigmoid),
                ],
            );
            print!("benchmarking {type_name} ...");
            std::io::stdout().flush().unwrap();
            let before = Instant::now();
            for i in 0..n_times {
                if i % (n_times / n_times.min(10)) == 0 || i == n_times - 1 {
                    let percentage = (i as f32) / (n_times as f32) * 100.0;
                    print!(" {percentage:.0}%");
                    stdout().flush().unwrap();
                }
                std::hint::black_box(nn.forward(&[0.0, 0.0, 0.0, 0.0]));
            }
            let after = Instant::now();
            let time = after.duration_since(before);
            let time_per_forward = time / n_times;
            println!(" ... {time:.4?} total, {time_per_forward:.0?} per forward");
        }};
    }

    println!("Test neural network: 4(inputs)*8*8*1, sigmoid activation");
    println!("Forward function {n_times} times");

    benchmark!(f16);
    benchmark!(f32);
    benchmark!(f64);
    // benchmark!(f128);
}
