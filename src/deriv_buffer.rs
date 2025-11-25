use std::{alloc::Allocator, iter, ptr::NonNull};

use faer::prelude::*;

use crate::{ColPtr, MatPtr, zeroed_box_slice_in};

#[derive(Clone, Copy, PartialEq, Eq)]
struct RawLayer {
    da_previous: Option<ColPtr<f32>>,
    dw: MatPtr<f32>,
    db: ColPtr<f32>,
    da: ColPtr<f32>,
}

impl RawLayer {
    unsafe fn as_layer_mut(&mut self) -> DerivLayerMut<'_> {
        unsafe {
            DerivLayerMut {
                da_previous: self.da_previous.map(|p| p.as_col_mut()),
                dw: self.dw.as_mat_mut(),
                db: self.db.as_col_mut(),
                da: self.da.as_col_ref(),
            }
        }
    }
}

#[derive(Debug)]
pub struct DerivLayerMut<'a> {
    /// dl_i / da^(u-1).
    pub da_previous: Option<ColMut<'a, f32>>,
    pub dw: MatMut<'a, f32>,
    pub db: ColMut<'a, f32>,
    /// dl_i / da^(u).
    /// For output layer case (u = V), this vector is irrelavent, it is nevertheless allocated for
    /// simplicity sake.
    pub da: ColRef<'a, f32>,
}

/// Buffer for performing gradient descent.
#[derive(Clone)]
pub struct NeuralNetworkDerivs<A: Allocator> {
    layers: Box<[RawLayer], A>,
    _buffer: Box<[f32], A>,
}

impl<A: Allocator> NeuralNetworkDerivs<A> {
    pub fn new_in(alloc: A, n_inputs: usize, layer_sizes: Vec<usize, A>) -> Self
    where
        A: Clone,
    {
        let n_floats = {
            let mut allocation_size = 0usize;
            let mut n_previous = n_inputs;
            for &n in &layer_sizes {
                allocation_size += Self::size_of_layer(n_previous, n);
                n_previous = n;
            }
            allocation_size
        };
        let n_layers = layer_sizes.len();
        let buffer: Box<[f32], A> = zeroed_box_slice_in(n_floats, alloc.clone());
        let layers: Box<[RawLayer], A> = {
            let mut layers = Box::new_uninit_slice_in(n_layers, alloc);
            let data: NonNull<f32> = NonNull::from(buffer.as_ref()).cast();
            let mut counter_params = 0usize;
            let mut n_previous = n_inputs;
            for (u, (layer, n)) in iter::zip(&mut layers, layer_sizes).enumerate() {
                let offset_dw = counter_params;
                let offset_db = offset_dw + n * n_previous;
                let offset_da = offset_db + n;
                let offset_da_previous = match u {
                    0 => None,
                    _ => Some(counter_params - n_previous),
                };
                counter_params = offset_da + n;
                let layer_ = unsafe {
                    RawLayer {
                        da_previous: offset_da_previous
                            .map(|offset| ColPtr::with_offset(data, offset, n_previous)),
                        dw: MatPtr::with_offset(data, offset_dw, n, n_previous),
                        db: ColPtr::with_offset(data, offset_db, n),
                        da: ColPtr::with_offset(data, offset_da, n),
                    }
                };
                layer.write(layer_);
                n_previous = n;
            }
            unsafe { layers.assume_init() }
        };
        Self {
            layers,
            _buffer: buffer,
        }
    }

    fn size_of_layer(n_previous: usize, n: usize) -> usize {
        n_previous * n // dw
            + n // db
            + n // da
    }

    /// Returns `None` if `i_layer` is not in `1..=n_layers`.
    pub fn layer_mut<'a, 'b, 'x>(&'a mut self, i_layer: usize) -> Option<DerivLayerMut<'x>>
    where
        'a: 'x,
        'b: 'x,
    {
        let raw_layer = self.layers.get_mut(i_layer.checked_sub(1)?)?;
        Some(unsafe { raw_layer.as_layer_mut() })
    }

    /// # Safety
    ///
    /// `i_layer` must be in `1..=n_layers`.
    pub unsafe fn layer_unchecked_mut<'a, 'b, 'x>(
        &'a mut self,
        i_layer: usize,
    ) -> DerivLayerMut<'x>
    where
        'a: 'x,
        'b: 'x,
    {
        debug_assert!((1..=self.layers.len()).contains(&i_layer));
        let raw_layer = unsafe { self.layers.get_unchecked_mut(i_layer.unchecked_sub(1)) };
        unsafe { raw_layer.as_layer_mut() }
    }
}
