use std::slice::GetDisjointMutError;

use faer::{col::AsColRef, linalg::matmul::matmul, prelude::*};

use crate::{
    ActivationFunction, DynActivationFunction, assume,
    buffers::{ParamBuffer, ResultBuffer},
    param_buffer, result_buffer,
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
            layer_descriptions: layer_descriptions.into(),
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

/// # Safety
///
/// - `param_buffer` and `result_buffer` must be of the same typology
/// - `input` must have the correct number of rows
pub unsafe fn forward_unchecked(
    input: ColRef<f32>,
    param_buffer: &ParamBuffer,
    result_buffer: &mut ResultBuffer,
) {
    // Safety: function's safety contract.
    unsafe { assume!(param_buffer.n_layers() == result_buffer.n_layers()) };
    for u in 0..param_buffer.n_layers() {
        let _layer_prev: result_buffer::LayerMut;
        let layer_params = param_buffer.layer(u).unwrap();
        let (a_prev, mut layer_results): (ColRef<f32>, result_buffer::LayerMut) =
            match u.checked_sub(1) {
                None => {
                    let layer_results = result_buffer.layer_mut(u).unwrap();
                    (input, layer_results)
                }
                Some(u_prev) => {
                    let [layer_prev_results, layer_results] =
                        unsafe { result_buffer.layer_disjoint_unchecked_mut([u_prev, u]) };
                    _layer_prev = layer_prev_results;
                    (_layer_prev.a.as_col_ref(), layer_results)
                }
            };
        let n_k = layer_params.n;
        let n_g = layer_params.n_previous;
        // Safety: function's safety contract.
        unsafe { assume!(a_prev.nrows() == n_g) };
        unsafe { assume!(layer_results.z.nrows() == n_k) }
        unsafe { assume!(layer_results.a.nrows() == n_k) }
        unsafe { assume!(layer_params.b.nrows() == n_k) }
        unsafe { assume!(layer_params.w.nrows() == n_k) }
        unsafe { assume!(layer_params.w.ncols() == n_g) }
        // z = W * a_prev;
        matmul(
            // A = α*L*R + β*A
            layer_results.z.rb_mut(), // A = z
            faer::Accum::Replace,     // β = 0.0
            layer_params.w,           // L = W
            a_prev,                   // R = a_prev
            1.0,                      // α = 1.0
            Par::Seq,
        );
        // z += b; a = phi(z);
        for k in 0..layer_params.n {
            layer_results.z[k] += layer_params.b[k];
            layer_results.a[k] = layer_params.phi.apply(layer_results.z[k]);
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
