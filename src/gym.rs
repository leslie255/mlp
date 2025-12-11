use std::{marker::PhantomData, ptr::NonNull, sync::mpsc};

use faer::ColRef;

use crate::{
    NeuralNetwork, Topology,
    core::{
        DerivBuffer, ParamBuffer, ResultBuffer, apply_derivs, calculate_derivs, forward_unchecked,
    },
};

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

    pub fn forward(&mut self, input: ColRef<f32>) -> ColRef<'_, f32> {
        let results = self
            .results
            .get_or_insert_with(|| ResultBuffer::create(&self.topology));
        let params: &'a mut ParamBuffer = unsafe { &mut *self.params.as_ptr() };
        assert!(input.nrows() == self.topology.n_inputs());
        unsafe { forward_unchecked(input, params, results) };
        results.layer(results.n_layers() - 1).unwrap().a
    }

    /// Returns the loss.
    pub fn train_single_threaded(&mut self, eta: f32, samples: &[f32]) -> f32 {
        assert!(!samples.is_empty());
        let params = unsafe { &mut *self.params.as_ptr() };
        self.results
            .get_or_insert_with(|| ResultBuffer::create(&self.topology));
        self.derivs
            .get_or_insert_with(|| DerivBuffer::create(&self.topology));
        let results = self.results.as_mut().unwrap();
        let derivs = self.derivs.as_mut().unwrap();
        let loss = unsafe { calculate_derivs(params, results, derivs, samples) };
        unsafe { apply_derivs(params, derivs, eta) };
        loss
    }

    /// Returns the loss.
    ///
    /// Calls `train_single_threaded` if `n_threads == 0`.
    pub fn train(&mut self, n_threads: usize, eta: f32, samples: &[f32]) -> f32 {
        if n_threads == 0 {
            return self.train_single_threaded(eta, samples);
        }
        let n_threads = n_threads.min(samples.len());
        let (n_inputs, n_outputs) = {
            let params = unsafe { self.params.as_ref() };
            let layer0 = params.layer(0).unwrap();
            let layer_last = params.layer(params.n_layers() - 1).unwrap();
            (layer0.n_previous, layer_last.n)
        };
        let sample_size = n_inputs + n_outputs;
        let chunk_size = samples.len() / sample_size / n_threads * sample_size;
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

fn worker(params: &ParamBuffer, topology: &Topology, samples: &[f32]) -> WorkerResult {
    let mut results = ResultBuffer::create(topology);
    let mut derivs = DerivBuffer::create(topology);
    let loss = unsafe { calculate_derivs(params, &mut results, &mut derivs, samples) };
    WorkerResult { loss, derivs }
}
