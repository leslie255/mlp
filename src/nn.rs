use std::slice::GetDisjointMutError;

use faer::prelude::*;

use crate::{
    ActivationFunction, DynActivationFunction,
    core::{ParamBuffer, ResultBuffer, forward_unchecked, param_buffer, result_buffer},
};

#[derive(Debug, Clone)]
pub struct Typology {
    n_inputs: usize,
    layer_descriptions: Vec<LayerDescription>,
}

impl Typology {
    pub fn new(n_inputs: usize, layer_descriptions: Vec<LayerDescription>) -> Self {
        Self {
            n_inputs,
            layer_descriptions,
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
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
    typology: Typology,
    params: ParamBuffer,
    results: ResultBuffer,
}

impl NeuralNetwork {
    pub fn new(typology: Typology) -> Self {
        let params = ParamBuffer::create(&typology);
        let results = ResultBuffer::create(&typology);
        // Safety: params and results are of the same typology as they are created from the same
        // n_inputs and layer_descriptions.
        unsafe { Self::from_raw_parts(typology, params, results) }
    }

    /// # Safety
    ///
    /// - `params` and `results` must be created from `typology`.
    pub unsafe fn from_raw_parts(
        typology: Typology,
        params: ParamBuffer,
        results: ResultBuffer,
    ) -> Self {
        Self {
            typology,
            params,
            results,
        }
    }

    pub fn into_raw_parts(self) -> (ParamBuffer, ResultBuffer) {
        (self.params, self.results)
    }

    pub fn forward(&mut self, input: ColRef<f32>) -> ColRef<'_, f32> {
        // Safety: params and results are created from the same typology.
        unsafe { forward_unchecked(input, &self.params, &mut self.results) };
        self.results.layer(self.results.n_layers() - 1).unwrap().a
    }

    pub fn typology(&self) -> &Typology {
        &self.typology
    }

    pub fn params(&self) -> &ParamBuffer {
        &self.params
    }

    pub fn params_layer(&self, index: usize) -> Option<param_buffer::LayerRef<'_>> {
        self.params().layer(index)
    }

    pub fn params_layer_mut(&mut self, index: usize) -> Option<param_buffer::LayerMut<'_>> {
        // Safety: typology cannot be changed by user when it only has access to a layer.
        unsafe { self.params_mut().layer_mut(index) }
    }

    pub fn params_layer_disjoint_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Result<[param_buffer::LayerMut<'_>; N], GetDisjointMutError> {
        // Safety: typology cannot be changed by user when it only has access to praticular layers.
        unsafe { self.params_mut().layer_disjoint_mut(indices) }
    }

    /// # Safety
    ///
    /// Typology of `params` must not be changed.
    pub unsafe fn params_mut(&mut self) -> &mut ParamBuffer {
        &mut self.params
    }

    pub fn results(&self) -> &ResultBuffer {
        &self.results
    }

    /// # Safety
    ///
    /// Typology of `results` must not be changed.
    pub unsafe fn results_mut(&mut self) -> &mut ResultBuffer {
        &mut self.results
    }

    pub fn results_layer(&self, index: usize) -> Option<result_buffer::LayerRef<'_>> {
        self.results().layer(index)
    }

    pub fn results_layer_mut(&mut self, index: usize) -> Option<result_buffer::LayerMut<'_>> {
        // Safety: typology cannot be changed by user when it only has access to a layer.
        unsafe { self.results_mut().layer_mut(index) }
    }

    pub fn results_layer_disjoint_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Result<[result_buffer::LayerMut<'_>; N], GetDisjointMutError> {
        // Safety: typology cannot be changed by user when it only has access to praticular layers.
        unsafe { self.results_mut().layer_disjoint_mut(indices) }
    }
}
