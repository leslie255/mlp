use std::{alloc::Allocator, slice};

use faer::prelude::*;

/// Buffer for performing gradient descent.
pub struct NeuralNetworkDerivs<A: Allocator> {
    n_inputs: usize,
    n_layers: usize,
    /// Number of neurons per layer.
    layer_sizes: Box<[usize], A>,
    /// The starting points of each layer in the buffer.
    /// (in f32 indices, not bytes).
    layer_offsets: Box<[usize], A>,
    buffer: Box<[f32], A>,
}

impl<A: Allocator> NeuralNetworkDerivs<A> {
    pub fn new_in(alloc: A, n_inputs: usize, layer_sizes: Vec<usize, A>) -> Self
    where
        A: Clone,
    {
        // Calculate allocation size and layers layout.
        let mut allocation_size = 0usize;
        let mut layer_offsets: Vec<usize, A> =
            Vec::with_capacity_in(layer_sizes.len(), alloc.clone());
        let mut n_previous = n_inputs; // number of neurons in the previous layer
        for &n in &layer_sizes {
            layer_offsets.push(allocation_size);
            allocation_size += n * n_previous; // weight
            allocation_size += n; // bias
            allocation_size += n; // da
            n_previous = n;
        }
        let buffer: Box<[f32], A> =
            unsafe { Box::new_zeroed_slice_in(allocation_size, alloc).assume_init() };
        Self {
            buffer,
            n_inputs,
            n_layers: layer_sizes.len(),
            layer_sizes: layer_sizes.into(),
            layer_offsets: layer_offsets.into(),
        }
    }

    /// # Safety
    ///
    /// - `offset..(offset + n_cols * n_rows)` must be within allocation.
    /// - No other references to this region exist, due to Rust's `&mut` aliasing rules.
    #[expect(dead_code)]
    unsafe fn mat_ref_unchecked(
        &self,
        offset: usize,
        n_cols: usize,
        n_rows: usize,
    ) -> MatRef<'_, f32> {
        let ptr: *const f32 = unsafe { self.buffer.as_ptr().cast::<f32>().add(offset) };
        MatRef::from_column_major_slice(
            unsafe { std::slice::from_raw_parts(ptr, n_cols * n_rows) },
            n_rows,
            n_cols,
        )
    }

    /// # Safety
    ///
    /// - `offset..(offset + n_cols * n_rows)` must be within allocation.
    /// - No other references to this region exist, due to Rust's `&mut` aliasing rules.
    unsafe fn mat_mut_unchecked(
        &self,
        offset: usize,
        n_cols: usize,
        n_rows: usize,
    ) -> MatMut<'_, f32> {
        let ptr: *mut f32 = unsafe { self.buffer.as_ptr().cast::<f32>().add(offset).cast_mut() };
        MatMut::from_column_major_slice_mut(
            unsafe { std::slice::from_raw_parts_mut(ptr, n_cols * n_rows) },
            n_rows,
            n_cols,
        )
    }

    /// # Safety
    ///
    /// - `offset..(offset + n)` must be within allocation.
    /// - No `&mut` references to this region exist, due to Rust's `&mut` aliasing rules.
    unsafe fn vec_ref_unchecked(&self, offset: usize, n: usize) -> ColRef<'_, f32> {
        let ptr: *const f32 = unsafe { self.buffer.as_ptr().cast::<f32>().add(offset) };
        ColRef::from_slice(unsafe { slice::from_raw_parts(ptr, n) })
    }

    /// # Safety
    ///
    /// - `offset..(offset + n)` must be within allocation.
    /// - No other references to this region exist, due to Rust's `&mut` aliasing rules.
    unsafe fn vec_mut_unchecked(&self, offset: usize, n: usize) -> ColMut<'_, f32> {
        let ptr: *mut f32 = unsafe { self.buffer.as_ptr().cast::<f32>().add(offset).cast_mut() };
        ColMut::from_slice_mut(unsafe { slice::from_raw_parts_mut(ptr, n) })
    }

    #[inline(always)]
    fn layer_size(&self, i_layer: usize) -> usize {
        match i_layer {
            0 => self.n_inputs,
            i_layer => self.layer_sizes[i_layer - 1],
        }
    }

    /// # Panics
    ///
    /// - if `i_layer == 0`.
    #[inline(always)]
    #[track_caller]
    fn layer_offset(&self, i_layer: usize) -> usize {
        match i_layer {
            0 => panic!(),
            i_layer => self.layer_offsets[i_layer - 1],
        }
    }

    /// Returns `None` if `i_layer` is not in `1..=N_LAYERS`.
    #[allow(unused_variables)]
    pub fn get_layer_mut<'a, 'b, 'x>(&'a mut self, u: usize) -> Option<DerivLayerMut<'x>>
    where
        'a: 'x,
        'b: 'x,
    {
        if !(1..=self.n_layers).contains(&u) {
            return None;
        }
        let n = self.layer_size(u);
        let n_previous = self.layer_size(u - 1);
        let offset_w = self.layer_offset(u);
        let offset_b = offset_w + n * n_previous;
        let offset_da = offset_b + n;
        let dw = unsafe { self.mat_mut_unchecked(offset_w, n_previous, n) };
        let db = unsafe { self.vec_mut_unchecked(offset_b, n) };
        let da_previous = if u != 1 {
            let offset_da_previous = offset_w - n_previous;
            Some(unsafe { self.vec_mut_unchecked(offset_da_previous, n_previous) })
        } else {
            None
        };
        let da = unsafe { self.vec_ref_unchecked(offset_da, n) };
        Some(DerivLayerMut {
            n_previous,
            n,
            da_previous,
            dw,
            db,
            da,
        })
    }
}

#[derive(Debug)]
pub struct DerivLayerMut<'a> {
    pub n_previous: usize,
    pub n: usize,
    /// dl_i / da^(u-1).
    pub da_previous: Option<ColMut<'a, f32>>,
    pub dw: MatMut<'a, f32>,
    pub db: ColMut<'a, f32>,
    /// dl_i / da^(u).
    /// For output layer case (u = V), this vector is irrelavent, it is nevertheless allocated for
    /// simplicity sake.
    pub da: ColRef<'a, f32>,
}
