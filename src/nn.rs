use std::{iter, slice::GetDisjointMutError};

use faer::prelude::*;
use rand::distr::uniform::SampleRange;

use crate::{
    ActivationFunction, DynActivationFunction,
    core::{ParamBuffer, ResultBuffer, forward_unchecked, param_buffer, result_buffer},
};

#[derive(Debug, Clone)]
pub struct Topology {
    n_inputs: usize,
    layer_descriptions: Vec<LayerDescription>,
}

impl Topology {
    pub fn new(n_inputs: usize, layer_descriptions: Vec<LayerDescription>) -> Self {
        Self {
            n_inputs,
            layer_descriptions,
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn n_outputs(&self) -> usize {
        self.layer_descriptions()
            .last()
            .map_or(self.n_inputs, |last_layer| last_layer.n_neurons)
    }

    pub fn layer_descriptions(&self) -> &[LayerDescription] {
        &self.layer_descriptions
    }

    pub fn n_layers(&self) -> usize {
        self.layer_descriptions().len()
    }
}

#[derive(Debug, Clone)]
pub struct LayerDescription {
    pub n_neurons: usize,
    pub phi: DynActivationFunction,
}

impl LayerDescription {
    pub fn new(n_neurons: usize, phi: impl ActivationFunction) -> Self {
        Self {
            n_neurons,
            phi: DynActivationFunction::new(phi),
        }
    }
}

pub struct NeuralNetwork {
    topology: Topology,
    params: ParamBuffer,
    results: ResultBuffer,
}

impl NeuralNetwork {
    pub fn new(topology: Topology) -> Self {
        let params = ParamBuffer::create(&topology);
        let results = ResultBuffer::create(&topology);
        // Safety: params and results are of the same topology as they are created from the same
        // n_inputs and layer_descriptions.
        unsafe { Self::from_raw_parts(topology, params, results) }
    }

    /// # Safety
    ///
    /// - `params` and `results` must be created from `topology`.
    pub unsafe fn from_raw_parts(
        topology: Topology,
        params: ParamBuffer,
        results: ResultBuffer,
    ) -> Self {
        Self {
            topology,
            params,
            results,
        }
    }

    pub fn into_raw_parts(self) -> (ParamBuffer, ResultBuffer) {
        (self.params, self.results)
    }

    pub fn n_inputs(&self) -> usize {
        self.topology().n_inputs()
    }

    pub fn n_outputs(&self) -> usize {
        self.topology().n_outputs()
    }

    pub fn forward(&mut self, input: ColRef<f32>) -> ColRef<'_, f32> {
        // Safety: params and results are created from the same topology.
        unsafe { forward_unchecked(input, &self.params, &mut self.results) };
        self.results.layer(self.results.n_layers() - 1).unwrap().a
    }

    pub fn loss(&mut self, samples: &[f32]) -> f32 {
        let mut loss = 0.0f32;
        let n_inputs = self.n_inputs();
        let n_outputs = self.n_outputs();
        for sample in samples.chunks(n_inputs + n_outputs) {
            let x = ColRef::from_slice(&sample[0..n_inputs]);
            let y = ColRef::from_slice(&sample[n_inputs..n_inputs + n_outputs]);
            let a = self.forward(x);
            loss += iter::zip(a.iter(), y.iter())
                .map(|(&ak, &yk)| (ak - yk).powi(2))
                .sum::<f32>();
        }
        loss
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn params(&self) -> &ParamBuffer {
        &self.params
    }

    /// # Safety
    ///
    /// Topology of `params` must not be changed.
    pub unsafe fn params_unchecked_mut(&mut self) -> &mut ParamBuffer {
        &mut self.params
    }

    pub fn params_as_slice(&self) -> &[f32] {
        self.params().as_slice()
    }

    pub fn params_as_mut_slice(&mut self) -> &mut [f32] {
        // Safety: topology cannot be changed by user when it only has access to params as a mut
        // slice.
        let params = unsafe { self.params_unchecked_mut() };
        params.as_mut_slice()
    }

    pub fn randomize_params(&mut self, range: impl SampleRange<f32> + Clone) {
        // Safety: param buffer topology is not changed.
        unsafe { self.params_unchecked_mut().randomize(range) };
    }

    pub fn params_layer(&self, index: usize) -> Option<param_buffer::LayerRef<'_>> {
        self.params().layer(index)
    }

    pub fn params_layer_mut(&mut self, index: usize) -> Option<param_buffer::LayerMut<'_>> {
        // Safety: topology cannot be changed by user when it only has access to a layer.
        unsafe { self.params_unchecked_mut().layer_mut(index) }
    }

    pub fn params_layer_disjoint_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Result<[param_buffer::LayerMut<'_>; N], GetDisjointMutError> {
        // Safety: topology cannot be changed by user when it only has access to praticular layers.
        unsafe { self.params_unchecked_mut().layer_disjoint_mut(indices) }
    }

    pub fn results(&self) -> &ResultBuffer {
        &self.results
    }

    /// # Safety
    ///
    /// Topology of `results` must not be changed.
    pub unsafe fn results_unchecked_mut(&mut self) -> &mut ResultBuffer {
        &mut self.results
    }

    pub fn results_layer(&self, index: usize) -> Option<result_buffer::LayerRef<'_>> {
        self.results().layer(index)
    }

    pub fn results_layer_mut(&mut self, index: usize) -> Option<result_buffer::LayerMut<'_>> {
        // Safety: topology cannot be changed by user when it only has access to a layer.
        unsafe { self.results_unchecked_mut().layer_mut(index) }
    }

    pub fn results_layer_disjoint_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Result<[result_buffer::LayerMut<'_>; N], GetDisjointMutError> {
        // Safety: topology cannot be changed by user when it only has access to praticular layers.
        unsafe { self.results_unchecked_mut().layer_disjoint_mut(indices) }
    }
}
