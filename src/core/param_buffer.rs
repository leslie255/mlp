use std::{array, iter, mem::transmute, ptr::NonNull, slice::GetDisjointMutError};

use faer::prelude::*;
use rand::{Rng, distr::uniform::SampleRange, rngs::ThreadRng};

use crate::{ColPtr, DynActivationFunction, MatPtr, PrettyPrintParams, Topology};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct LayerRaw {
    pub(crate) n: usize,
    pub(crate) n_previous: usize,
    pub(crate) w: MatPtr<f32>,
    pub(crate) b: ColPtr<f32>,
    pub(crate) phi: DynActivationFunction,
}

impl LayerRaw {
    /// # Safety
    ///
    /// - must satisfy aliasing rules of `&` references
    pub(crate) unsafe fn as_ref<'a>(self) -> LayerRef<'a> {
        unsafe { transmute(self) }
    }

    /// # Safety
    ///
    /// - must satisfy aliasing rules of `&mut` references
    pub(crate) unsafe fn as_mut<'a>(self) -> LayerMut<'a> {
        unsafe { transmute(self) }
    }
}

/// Immutable view of a layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerRef<'a> {
    /// Number of neurons in this layer.
    pub n: usize,
    /// Number of neurons in the previous layer.
    pub n_previous: usize,
    pub w: MatRef<'a, f32>,
    pub b: ColRef<'a, f32>,
    pub phi: DynActivationFunction,
}

/// Mutable view of a layer.
#[derive(Debug)]
pub struct LayerMut<'a> {
    /// Number of neurons in this layer.
    pub n: usize,
    /// Number of neurons in the previous layer.
    pub n_previous: usize,
    pub w: MatMut<'a, f32>,
    pub b: ColMut<'a, f32>,
    pub phi: DynActivationFunction,
}

/// Buffer for storing neural network parameters.
pub struct ParamBuffer {
    layers: Box<[LayerRaw]>,
    buffer: Box<[f32]>,
}

unsafe impl Send for ParamBuffer {}
unsafe impl Sync for ParamBuffer {}

impl ParamBuffer {
    pub fn create(topology: &Topology) -> Self {
        let n_floats = {
            let mut n_floats = 0usize;
            let mut n_previous = topology.n_inputs();
            for layer_description in topology.layer_descriptions() {
                let n = layer_description.n_neurons;
                n_floats += n * n_previous; // w
                n_floats += n; // b
                n_previous = n;
            }
            n_floats
        };
        assert!(n_floats != 0);
        let buffer: Box<[f32]> = bytemuck::zeroed_slice_box(n_floats);
        let buffer_ptr = NonNull::from_ref(&buffer[0]);
        let layers: Box<[LayerRaw]> = unsafe {
            let mut layers = Box::new_uninit_slice(topology.layer_descriptions().len());
            let mut n_previous = topology.n_inputs();
            let mut counter = 0usize;
            for (layer, layer_description) in
                iter::zip(&mut layers[..], topology.layer_descriptions())
            {
                let n = layer_description.n_neurons;
                let offset_w = counter;
                let offset_b = counter + n * n_previous;
                counter = offset_b + n;
                // Safety: offset_w, offset_b < buffer.len(), so we're offseting within the buffer.
                layer.write(LayerRaw {
                    n,
                    n_previous,
                    w: MatPtr::with_offset(buffer_ptr, offset_w, n, n_previous),
                    b: ColPtr::with_offset(buffer_ptr, offset_b, n),
                    phi: layer_description.phi,
                });
                n_previous = n;
            }
            // Safety: all layers are initialized in the loop above.
            layers.assume_init()
        };
        Self { layers, buffer }
    }

    pub fn randomize(&mut self, range: impl SampleRange<f32> + Clone) {
        let mut rng = ThreadRng::default();
        for p in self.as_mut_slice() {
            *p = rng.random_range(range.clone());
        }
    }

    pub fn pretty_print_layer(&self, index: usize) -> Option<PrettyPrintParams<'_>> {
        let layer = self.layer(index)?;
        Some(PrettyPrintParams::new(index, layer))
    }

    /// Direct access to the underlying buffer.
    /// Useful for dumping/loading params from file.
    pub fn as_slice(&self) -> &[f32] {
        &self.buffer
    }

    /// Direct access to the underlying buffer.
    /// Useful for dumping/loading params from file.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.buffer
    }

    /// Number of layers in the neural network.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// # Safety
    ///
    /// - `index` must be in range.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn layer_unchecked(&self, index: usize) -> LayerRef<'_> {
        debug_assert!(index < self.n_layers());
        // Safety: function's safety contract.
        let layer_raw = unsafe { self.layers.get_unchecked(index) };
        // Safety: self would be & borrowed for the duration that the layer lives outside.
        unsafe { layer_raw.as_ref() }
    }

    /// # Safety
    ///
    /// - `index` must be in range.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn layer_unchecked_mut(&mut self, index: usize) -> LayerMut<'_> {
        debug_assert!(index < self.n_layers());
        // Safety: function's safety contract.
        let layer_raw = unsafe { self.layers.get_unchecked(index) };
        // Safety: self would be &mut borrowed for the duration that layer lives outside.
        unsafe { layer_raw.as_mut() }
    }

    /// Get a immutable view of a layer.
    /// Returns `None` if `index` is out of range.
    #[track_caller]
    pub fn layer(&self, index: usize) -> Option<LayerRef<'_>> {
        if index < self.n_layers() {
            // Safety: function's safety contract.
            Some(unsafe { self.layer_unchecked(index) })
        } else {
            None
        }
    }

    /// Get a mutable view of a layer.
    /// Returns `None` if `index` is out of range.
    #[track_caller]
    pub fn layer_mut(&mut self, index: usize) -> Option<LayerMut<'_>> {
        if index < self.n_layers() {
            // Safety: function's safety contract.
            Some(unsafe { self.layer_unchecked_mut(index) })
        } else {
            None
        }
    }

    /// Get mutable views to multiple different layers.
    ///
    /// `indices` must be in ascending order.
    #[track_caller]
    pub fn layer_disjoint_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> Result<[LayerMut<'_>; N], GetDisjointMutError> {
        let layers_raw = self.layers.get_disjoint_mut(indices)?;
        // Safety:
        // - self would be &mut borrowed for the duration that layer lives outside.
        // - indices are unique, ensured by `get_disjoint_mut`.
        let layers: [LayerMut; N] = array::from_fn(|i| unsafe { layers_raw[i].as_mut() });
        Ok(layers)
    }

    /// # Safety
    ///
    /// - each value in `indices` must be unique, ensuring no two layers are `&mut` borrowed at the
    ///   same time
    /// - every value in `indices` must be in range
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub unsafe fn layer_disjoint_unchecked_mut<const N: usize>(
        &mut self,
        indices: [usize; N],
    ) -> [LayerMut<'_>; N] {
        let layers: [LayerMut; N] = array::from_fn(|i| unsafe {
            let index = indices[i];
            debug_assert!(index < self.layers.len());
            // Safety: function's safety contract.
            let layer_raw = self.layers.get_unchecked_mut(index);
            // Safety: self would be &mut borrowed for the duration that layer lives outside.
            layer_raw.as_mut()
        });
        layers
    }
}
