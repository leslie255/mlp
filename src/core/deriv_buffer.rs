use std::{array, iter, mem::transmute, ptr::NonNull, slice::GetDisjointMutError};

use faer::prelude::*;

use crate::{ColPtr, MatPtr, PrettyPrintDerivs, Topology};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct LayerRaw {
    pub(crate) n: usize,
    pub(crate) n_previous: usize,
    pub(crate) dw: MatPtr<f32>,
    pub(crate) db: ColPtr<f32>,
    pub(crate) da: ColPtr<f32>,
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
    /// Short for `\frac{\partial L}{\partial W}` aka "dL/dW", where `L` is the loss over the
    /// training samples.
    pub dw: MatRef<'a, f32>,
    /// Short for `\frac{\partial L}{\partial b}` aka "dL/dW", where `L` is the loss over the
    /// training samples.
    pub db: ColRef<'a, f32>,
    /// Short for `\frac{\partial l_i}{\partial b}` aka "dL/dW", where `l_i` is the loss over one
    /// training sample.
    /// Needs zeroing per-sample, unlike `dw` and `db`.
    pub da: ColRef<'a, f32>,
}

/// Mutable view of a layer.
#[derive(Debug)]
pub struct LayerMut<'a> {
    /// Number of neurons in this layer.
    pub n: usize,
    /// Number of neurons in the previous layer.
    pub n_previous: usize,
    /// Short for `\frac{\partial L}{\partial W}` aka "dL/dW", where `L` is the loss over the
    /// training samples.
    pub dw: MatMut<'a, f32>,
    /// Short for `\frac{\partial L}{\partial b}` aka "dL/dW", where `L` is the loss over the
    /// training samples.
    pub db: ColMut<'a, f32>,
    /// Short for `\frac{\partial l_i}{\partial b}` aka "dL/dW", where `l_i` is the loss over one
    /// training sample.
    /// Needs zeroing per-sample, unlike `dw` and `db`.
    pub da: ColMut<'a, f32>,
}

/// Buffer needed for performing back propagation on neural network.
pub struct DerivBuffer {
    layers: Box<[LayerRaw]>,
    da_start: usize,
    buffer: Box<[f32]>,
}

unsafe impl Send for DerivBuffer {}
unsafe impl Sync for DerivBuffer {}

impl DerivBuffer {
    pub fn create(topology: &Topology) -> Self {
        let (n_floats, da_start) = {
            let mut n_floats = 0usize;
            let mut da_start = 0usize;
            let mut n_previous = topology.n_inputs();
            for layer_description in topology.layer_descriptions() {
                let n = layer_description.n_neurons;
                let dw_size = n * n_previous;
                let db_size = n;
                let da_size = n;
                n_floats += dw_size; // dw
                n_floats += db_size; // db
                n_floats += da_size; // da
                da_start += dw_size; // db
                da_start += db_size; // db
                n_previous = n;
            }
            (n_floats, da_start)
        };
        assert!(n_floats != 0);
        let buffer: Box<[f32]> = bytemuck::zeroed_slice_box(n_floats);
        let buffer_ptr = NonNull::from_ref(&buffer[0]);
        let layers: Box<[LayerRaw]> = unsafe {
            let mut layers = Box::new_uninit_slice(topology.layer_descriptions().len());
            let mut n_previous = topology.n_inputs();
            let mut counter_params = 0usize;
            let mut counter_da = da_start;
            for (layer, layer_description) in
                iter::zip(&mut layers[..], topology.layer_descriptions())
            {
                let n = layer_description.n_neurons;
                let offset_dw = counter_params;
                let offset_db = counter_params + n * n_previous;
                counter_params = offset_db + n;
                let offset_da = counter_da;
                counter_da += n;
                // Safety: offset_w, offset_b < buffer.len(), so we're offseting within the buffer.
                layer.write(LayerRaw {
                    n,
                    n_previous,
                    dw: MatPtr::with_offset(buffer_ptr, offset_dw, n, n_previous),
                    db: ColPtr::with_offset(buffer_ptr, offset_db, n),
                    da: ColPtr::with_offset(buffer_ptr, offset_da, n),
                });
                n_previous = n;
            }
            // Safety: all layers are initialized in the loop above.
            layers.assume_init()
        };
        Self {
            layers,
            da_start,
            buffer,
        }
    }

    /// Zero all the `dw` and `db`s.
    pub(crate) fn clear_params(&mut self) {
        bytemuck::fill_zeroes(self.params_mut());
    }

    /// Number of layers in the neural network.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn pretty_print_layer(&self, index: usize) -> Option<PrettyPrintDerivs<'_>> {
        let layer = self.layer(index)?;
        Some(PrettyPrintDerivs::new(index, layer))
    }

    /// `&mut` reference to the params section (storage of `dw` and `db`s) of the buffer.
    pub(crate) fn params(&self) -> &[f32] {
        &self.buffer[0..self.da_start]
    }

    /// `&mut` reference to the params section (storage of `dw` and `db`s) of the buffer.
    pub(crate) fn params_mut(&mut self) -> &mut [f32] {
        &mut self.buffer[0..self.da_start]
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
