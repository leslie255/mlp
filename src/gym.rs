use std::{marker::PhantomData, ptr::NonNull, sync::mpsc};

use crate::{
    NeuralNetwork, Topology,
    core::{DerivBuffer, ParamBuffer, ResultBuffer, apply_derivs, calculate_derivs},
};

/// Calculates and applies derivative.
///
/// Returns loss over the provided samples.
///
/// # Safety
///
/// - `param_buffer`, `result_buffer` and `deriv_buffer` must be of the same topology
/// - all inputs and outputs in `samples` must be of the correct sizes
pub unsafe fn train_single_threaded<'a>(
    eta: f32,
    param_buffer: &mut ParamBuffer,
    result_buffer: &mut ResultBuffer,
    deriv_buffer: &mut DerivBuffer,
    samples: impl IntoIterator<Item = &'a (&'a [f32], &'a [f32])>,
) -> f32 {
    let loss = unsafe { calculate_derivs(param_buffer, result_buffer, deriv_buffer, samples) };
    unsafe { apply_derivs(param_buffer, deriv_buffer, eta) };
    loss
}

pub struct Gym<'a> {
    topology: Topology,
    params: NonNull<ParamBuffer>,
    results: Option<ResultBuffer>,
    derivs: Option<DerivBuffer>,
    _marker: PhantomData<&'a mut ParamBuffer>,
}

struct WorkerResult {
    loss: f32,
    derivs: DerivBuffer,
}

impl<'a> Gym<'a> {
    pub fn new(nn: &'a mut NeuralNetwork) -> Self {
        Self {
            topology: nn.topology().clone(),
            params: unsafe { NonNull::from_mut(nn.params_unchecked_mut()) },
            results: None,
            derivs: None,
            _marker: PhantomData,
        }
    }

    pub fn train_single_threaded(&mut self, eta: f32, samples: &[(&[f32], &[f32])]) -> f32 {
        assert!(!samples.is_empty());
        unsafe {
            let params = &mut *self.params.as_ptr();
            self.results
                .get_or_insert_with(|| ResultBuffer::create(&self.topology));
            self.derivs
                .get_or_insert_with(|| DerivBuffer::create(&self.topology));
            train_single_threaded(
                eta,
                params,
                self.results.as_mut().unwrap(),
                self.derivs.as_mut().unwrap(),
                samples,
            )
        }
    }

    pub fn train(&mut self, n_threads: usize, eta: f32, samples: &[(&[f32], &[f32])]) -> f32 {
        if n_threads == 0 {
            return self.train_single_threaded(eta, samples);
        }
        let n_threads = n_threads.min(samples.len());
        let chunk_size = samples.len() / n_threads;
        let (tx, rx) = mpsc::channel();
        std::thread::scope(|s| {
            for i in 0..n_threads {
                let tx = tx.clone();
                let is_last = i + 1 == n_threads;
                let samples_chunk = match is_last {
                    true => &samples[i * chunk_size..],
                    false => &samples[i * chunk_size..(i + 1) * chunk_size],
                };
                let params = unsafe { &*self.params.as_ptr() };
                let topology = &self.topology;
                s.spawn(move || {
                    let result = worker(params, topology, samples_chunk);
                    tx.send(result).unwrap();
                });
            }
        });
        let mut loss = 0.0f32;
        for _ in 0..n_threads {
            let result = rx.recv().unwrap();
            loss += result.loss;
            let params = unsafe { &mut *self.params.as_ptr() };
            unsafe { apply_derivs(params, &result.derivs, eta) };
        }
        loss / (n_threads as f32)
    }
}

fn worker(params: &ParamBuffer, topology: &Topology, samples: &[(&[f32], &[f32])]) -> WorkerResult {
    let mut results = ResultBuffer::create(topology);
    let mut derivs = DerivBuffer::create(topology);
    let loss = unsafe { calculate_derivs(params, &mut results, &mut derivs, samples) };
    WorkerResult { loss, derivs }
}
