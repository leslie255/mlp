use std::{array, iter, mem::transmute, ptr::NonNull, slice::GetDisjointMutError};

use faer::prelude::*;

use crate::{ColPtr, Topology};

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct LayerRaw {
    pub(crate) n: usize,
    pub(crate) n_previous: usize,
    pub(crate) z: ColPtr<f32>,
    pub(crate) a: ColPtr<f32>,
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
    pub z: ColRef<'a, f32>,
    pub a: ColRef<'a, f32>,
}

/// Mutable view of a layer.
#[derive(Debug)]
pub struct LayerMut<'a> {
    /// Number of neurons in this layer.
    pub n: usize,
    /// Number of neurons in the previous layer.
    pub n_previous: usize,
    pub z: ColMut<'a, f32>,
    pub a: ColMut<'a, f32>,
}

/// Buffer for storing neural network activation results.
pub struct ResultBuffer {
    layers: Box<[LayerRaw]>,
    _buffer: Box<[f32]>,
}

unsafe impl Send for ResultBuffer {}
unsafe impl Sync for ResultBuffer {}

impl ResultBuffer {
    pub fn create(topology: &Topology) -> Self {
        let n_floats = {
            let mut n_floats = 0usize;
            for layer_description in topology.layer_descriptions() {
                let n = layer_description.n_neurons;
                n_floats += n; // z
                n_floats += n; // a
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
                let offset_z = counter;
                let offset_a = counter + n;
                counter = offset_a + n;
                debug_assert!(offset_z + n <= buffer.len());
                debug_assert!(offset_a + n <= buffer.len());
                // Safety: offset_w, offset_b < buffer.len(), so we're offseting within the buffer.
                layer.write(LayerRaw {
                    n,
                    n_previous,
                    z: ColPtr::with_offset(buffer_ptr, offset_z, n),
                    a: ColPtr::with_offset(buffer_ptr, offset_a, n),
                });
                n_previous = n;
            }
            // Safety: all layers are initialized in the loop above.
            layers.assume_init()
        };
        Self {
            layers,
            _buffer: buffer,
        }
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
