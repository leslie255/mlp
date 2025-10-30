use crate::{activation::ActivationFunction, pretty_print::PrettyPrintParams};

use std::{
    alloc::{self, Allocator, Global},
    fmt::{self, Debug},
    ptr::{NonNull, drop_in_place},
    slice,
};

use faer::{Accum, linalg::matmul::matmul, prelude::*};

use rand::{Rng, distr::uniform::SampleRange, rngs::ThreadRng};

pub struct LayerDescription {
    pub n_neurons: usize,
    pub phi: Box<dyn ActivationFunction>,
}

impl LayerDescription {
    pub fn new(n_neurons: usize, phi: impl ActivationFunction + 'static) -> Self {
        Self {
            n_neurons,
            phi: Box::new(phi),
        }
    }
}

impl Debug for LayerDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerDescription")
            .field("n_neurons", &self.n_neurons)
            .field("phi", &self.phi.name())
            .finish()
    }
}

pub struct NeuralNetwork<const N_LAYERS: usize, A: Allocator> {
    allocator: A,
    allocation_ptr: NonNull<u8>,
    /// Number of `f32`'s allocated.
    allocation_size: usize,
    n_inputs: usize,
    layer_descriptions: [LayerDescription; N_LAYERS],
    /// The starting points of each layer in the buffer.
    /// (in f32 indices, not bytes).
    layer_offsets: [usize; N_LAYERS],
}

impl<const N_LAYERS: usize, A: Allocator> Drop for NeuralNetwork<N_LAYERS, A> {
    fn drop(&mut self) {
        unsafe {
            let allocation_layout = Self::allocation_layout(self.allocation_size);
            self.allocator
                .deallocate(self.allocation_ptr, allocation_layout);
            drop_in_place(&mut self.allocator);
            drop_in_place(&mut self.layer_descriptions);
        }
    }
}

impl<const N_LAYERS: usize> NeuralNetwork<N_LAYERS, Global> {
    pub fn new(n_inputs: usize, architecture: [LayerDescription; N_LAYERS]) -> Self {
        Self::new_in(Global, n_inputs, architecture)
    }
}

impl<const N_LAYERS: usize, A: Allocator> NeuralNetwork<N_LAYERS, A> {
    pub fn new_in(
        allocator: A,
        n_inputs: usize,
        layer_descriptions: [LayerDescription; N_LAYERS],
    ) -> Self {
        // Calculate allocation size and layers layout.
        let mut allocation_size = 0usize;
        let mut layer_offsets = [0usize; N_LAYERS];
        let mut n_previous = n_inputs; // number of neurons in the previous layer
        for (i_layer, layer_description) in layer_descriptions.iter().enumerate() {
            layer_offsets[i_layer] = allocation_size;
            let n = layer_description.n_neurons;
            allocation_size += n * n_previous; // W
            allocation_size += n * 3;
            n_previous = n;
        }
        let layout = Self::allocation_layout(allocation_size);
        let allocation_ptr = allocator.allocate_zeroed(layout).unwrap();
        Self {
            allocator,
            allocation_ptr: allocation_ptr.cast(),
            allocation_size,
            n_inputs,
            layer_descriptions,
            layer_offsets,
        }
    }

    fn allocation_layout(n_floats: usize) -> alloc::Layout {
        alloc::Layout::array::<f32>(n_floats).unwrap()
    }

    pub fn n_layers(&self) -> usize {
        N_LAYERS
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
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
        let ptr: *mut f32 = unsafe { self.allocation_ptr.as_ptr().cast::<f32>().add(offset) };
        MatMut::from_column_major_slice_mut(
            unsafe { std::slice::from_raw_parts_mut(ptr, n_cols * n_rows) },
            n_rows,
            n_cols,
        )
    }

    /// # Safety
    ///
    /// - `offset..(offset + n_cols * n_rows)` must be within allocation.
    /// - No other references to this region exist, due to Rust's `&mut` aliasing rules.
    unsafe fn mat_ref_unchecked(
        &self,
        offset: usize,
        n_cols: usize,
        n_rows: usize,
    ) -> MatRef<'_, f32> {
        let ptr: *const f32 = unsafe { self.allocation_ptr.as_ptr().cast::<f32>().add(offset) };
        MatRef::from_column_major_slice(
            unsafe { std::slice::from_raw_parts(ptr, n_cols * n_rows) },
            n_rows,
            n_cols,
        )
    }

    /// # Safety
    ///
    /// - `offset..(offset + n)` must be within allocation.
    /// - No other references to this region exist, due to Rust's `&mut` aliasing rules.
    unsafe fn vec_mut_unchecked(&self, offset: usize, n: usize) -> ColMut<'_, f32> {
        let ptr: *mut f32 = unsafe { self.allocation_ptr.as_ptr().cast::<f32>().add(offset) };
        ColMut::from_slice_mut(unsafe { slice::from_raw_parts_mut(ptr, n) })
    }

    /// # Safety
    ///
    /// - `offset..(offset + n)` must be within allocation.
    /// - No `&mut` references to this region exist, due to Rust's `&mut` aliasing rules.
    unsafe fn vec_ref_unchecked(&self, offset: usize, n: usize) -> ColRef<'_, f32> {
        let ptr: *mut f32 = unsafe { self.allocation_ptr.as_ptr().cast::<f32>().add(offset) };
        ColRef::from_slice(unsafe { slice::from_raw_parts_mut(ptr, n) })
    }

    /// Returns `None` if `i_layer` is not in `1..=N_LAYERS`.
    /// The `a_previous` field of the returned `LayerMut` will be `None` if both:
    /// - `i_layer == 1`, and
    /// - `input_layer` is `None`
    #[allow(unused_variables)]
    pub fn get_layer_mut<'a, 'b, 'x>(
        &'a mut self,
        input_layer: Option<ColRef<'b, f32>>,
        i_layer: usize,
    ) -> Option<LayerMut<'x>>
    where
        'a: 'x,
        'b: 'x,
    {
        if !(1..=N_LAYERS).contains(&i_layer) {
            return None;
        }
        let phi = self.layer_descriptions[i_layer - 1].phi.as_ref();
        let n = self.layer_descriptions[i_layer - 1].n_neurons;
        let n_previous = match i_layer {
            1 => self.n_inputs,
            i_layer => self.layer_descriptions[i_layer - 2].n_neurons,
        };
        let i_w = self.layer_offsets[i_layer - 1];
        let i_b = i_w + n * n_previous;
        let i_z = i_b + n;
        let i_a = i_z + n;
        let w = unsafe { self.mat_mut_unchecked(i_w, n_previous, n) };
        let b = unsafe { self.vec_mut_unchecked(i_b, n) };
        let z = unsafe { self.vec_mut_unchecked(i_z, n) };
        let a = unsafe { self.vec_mut_unchecked(i_a, n) };
        let a_previous = match i_layer {
            1 => input_layer,
            i_layer => {
                let i_a_previous = i_w - n_previous;
                Some(unsafe { self.vec_ref_unchecked(i_a_previous, n_previous) })
            }
        };
        Some(LayerMut {
            n_previous,
            n,
            a_previous,
            phi,
            w,
            b,
            z,
            a,
        })
    }

    pub fn randomize(
        &mut self,
        w_range: impl SampleRange<f32> + Clone,
        b_range: impl SampleRange<f32> + Clone,
    ) {
        let mut rng = ThreadRng::default();
        for i_layer in 1..=N_LAYERS {
            let layer = self.get_layer_mut(None, i_layer).unwrap();
            for column in layer.w.col_iter_mut() {
                for element in column.iter_mut() {
                    *element = rng.random_range(w_range.clone());
                }
            }
            for element in layer.b.iter_mut() {
                *element = rng.random_range(b_range.clone());
            }
        }
    }

    pub fn forward<'a>(&'a mut self, input_layer: ColRef<f32>) -> ColMut<'a, f32> {
        if input_layer.nrows() != self.n_inputs() {
            panic!(
                "Neural network is provided incorrect input dimensions (expected {}, found {})",
                self.n_inputs(),
                input_layer.nrows()
            );
        }
        for i_layer in 1..=N_LAYERS {
            let mut layer = self.get_layer_mut(Some(input_layer), i_layer).unwrap();
            matmul(
                layer.z.rb_mut(),
                Accum::Replace,
                layer.w.rb_mut(),
                layer.a_previous.unwrap().rb_mut(),
                1.0,
                Par::Seq,
            );
            zip!(&mut layer.a, &layer.z, &layer.b)
                .for_each(|unzip!(a, z, b)| *a = layer.phi.apply(*z + *b));
        }
        self.get_layer_mut(None, N_LAYERS).unwrap().a
    }

    pub fn get_vector(&self, i_layer: usize, kind: VectorObject) -> Option<ColRef<'_, f32>> {
        if !(1..=N_LAYERS).contains(&i_layer) {
            return None;
        }
        let n = self.layer_descriptions[i_layer - 1].n_neurons;
        let n_previous = match i_layer {
            1 => self.n_inputs,
            i_layer => self.layer_descriptions[i_layer - 2].n_neurons,
        };
        let layer_offset = self.layer_offsets[i_layer - 1];
        let offset = layer_offset + kind.suboffset(n, n_previous);
        Some(unsafe { self.vec_ref_unchecked(offset, n) })
    }

    pub fn get_activation(&self, i_layer: usize) -> Option<ColRef<'_, f32>> {
        self.get_vector(i_layer, VectorObject::Activation)
    }

    pub fn get_pre_activation(&self, i_layer: usize) -> Option<ColRef<'_, f32>> {
        self.get_vector(i_layer, VectorObject::PreActivation)
    }

    pub fn get_bias(&self, i_layer: usize) -> Option<ColRef<'_, f32>> {
        self.get_vector(i_layer, VectorObject::Bias)
    }

    pub fn get_weight(&self, i_layer: usize) -> Option<MatRef<'_, f32>> {
        if !(1..=N_LAYERS).contains(&i_layer) {
            return None;
        }
        let n = self.layer_descriptions[i_layer - 1].n_neurons;
        let n_previous = match i_layer {
            1 => self.n_inputs,
            i_layer => self.layer_descriptions[i_layer - 2].n_neurons,
        };
        let layer_offset = self.layer_offsets[i_layer - 1];
        Some(unsafe { self.mat_ref_unchecked(layer_offset, n_previous, n) })
    }
}

pub struct LayerMut<'a> {
    pub n_previous: usize,
    pub n: usize,
    pub a_previous: Option<ColRef<'a, f32>>,
    pub phi: &'a dyn ActivationFunction,
    pub w: MatMut<'a, f32>,
    pub b: ColMut<'a, f32>,
    pub z: ColMut<'a, f32>,
    pub a: ColMut<'a, f32>,
}

impl Debug for LayerMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LayerMut")
            .field("n_previous", &self.n_previous)
            .field("n", &self.n)
            .field("a_previous", &self.a_previous)
            .field("phi", &self.phi.name())
            .field("w", &self.w)
            .field("b", &self.b)
            .field("z", &self.z)
            .field("a", &self.a)
            .finish()
    }
}

impl<'a> LayerMut<'a> {
    pub fn pretty_print_params(&'a self, i_layer: usize) -> PrettyPrintParams<'a> {
        PrettyPrintParams::new(i_layer, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorObject {
    Bias,
    PreActivation,
    Activation,
}

impl VectorObject {
    const fn suboffset(self, n: usize, n_previous: usize) -> usize {
        let base_suboffset = n * n_previous; // weight matrix
        match self {
            VectorObject::Bias => base_suboffset,
            VectorObject::PreActivation => base_suboffset + n,
            VectorObject::Activation => base_suboffset + n + n,
        }
    }
}
