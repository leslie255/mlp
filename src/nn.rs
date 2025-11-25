use std::{
    alloc::{Allocator, Global},
    fmt::{self, Debug},
    iter,
    ptr::{NonNull, copy_nonoverlapping, write_bytes},
};

use faer::{Accum, linalg::matmul::matmul, prelude::*};

use rand::{Rng, distr::uniform::SampleRange, rngs::ThreadRng};

use derive_more::{Display, Error};

use crate::{ActivationFunction, ColPtr, Gym, MatPtr, NeuralNetworkDerivs, PrettyPrintParams};

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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LayerDescription")
            .field("n_neurons", &self.n_neurons)
            .field("phi", &self.phi.name())
            .finish()
    }
}

struct RawLayer {
    n_previous: usize,
    n: usize,
    w: MatPtr<f32>,
    b: ColPtr<f32>,
    phi: Box<dyn ActivationFunction>,
    z: ColPtr<f32>,
    a: ColPtr<f32>,
}

impl RawLayer {
    unsafe fn as_layer_mut(&mut self) -> LayerMut<'_> {
        unsafe {
            LayerMut {
                n_previous: self.n_previous,
                n: self.n,
                phi: &*self.phi,
                w: self.w.as_mat_mut(),
                b: self.b.as_col_mut(),
                z: self.z.as_col_mut(),
                a: self.a.as_col_mut(),
            }
        }
    }
}

struct Storage<A: Allocator> {
    layers: Box<[RawLayer], A>,
    buffer: Box<[f32], A>,
    /// Start of the region in `buffer` where storage of activation results (z and a) begin.
    za_start: usize,
}

#[derive(Debug, Clone, Display, Error)]
pub enum LoadParamsError {
    #[display("incorrect length")]
    IncorrectLength,
}

impl<A: Allocator> Storage<A> {
    fn new_in<const N_LAYERS: usize>(
        alloc: A,
        n_inputs: usize,
        layer_descriptions: [LayerDescription; N_LAYERS],
    ) -> Self
    where
        A: Clone,
    {
        let (n_floats, za_start) = {
            let mut allocation_size = 0usize;
            let mut za_start = 0usize;
            let mut n_previous = n_inputs;
            for layer_description in &layer_descriptions {
                let n = layer_description.n_neurons;
                allocation_size += Self::size_of_layer(n_previous, n);
                za_start += Self::size_of_layer_params(n_previous, n);
                n_previous = n;
            }
            (allocation_size, za_start)
        };
        let n_layers = layer_descriptions.len();
        let buffer: Box<[f32], A> = {
            let mut buffer = Box::new_uninit_slice_in(n_floats, alloc.clone());
            bytemuck::fill_zeroes(&mut buffer);
            unsafe {
                write_bytes(buffer.as_mut_ptr(), 0u8, n_floats);
                buffer.assume_init()
            }
        };
        let layers: Box<[RawLayer], A> = {
            let mut layers = Box::new_uninit_slice_in(n_layers, alloc);
            let data: NonNull<f32> = NonNull::from(buffer.as_ref()).cast();
            let mut counter_params = 0usize;
            let mut counter_za = za_start;
            let mut n_previous = n_inputs;
            for (layer, layer_description) in iter::zip(&mut layers, layer_descriptions) {
                let n = layer_description.n_neurons;
                let offset_w = counter_params;
                let offset_b = offset_w + n * n_previous;
                counter_params = offset_b + n;
                let offset_z = counter_za;
                let offset_a = counter_za + n;
                counter_za = offset_a + n;
                let layer_ = unsafe {
                    RawLayer {
                        n_previous,
                        n,
                        w: MatPtr::with_offset(data, offset_w, n, n_previous),
                        b: ColPtr::with_offset(data, offset_b, n),
                        phi: layer_description.phi,
                        z: ColPtr::with_offset(data, offset_z, n),
                        a: ColPtr::with_offset(data, offset_a, n),
                    }
                };
                layer.write(layer_);
                n_previous = n;
            }
            unsafe { layers.assume_init() }
        };
        Self {
            layers,
            buffer,
            za_start,
        }
    }

    fn size_of_layer_params(n_previous: usize, n: usize) -> usize {
        n_previous * n // w
            + n // b
    }

    fn size_of_layer(n_previous: usize, n: usize) -> usize {
        n_previous * n // w
            + n // b
            + n // a
            + n // z
    }

    fn raw_layer(&self, i_layer: usize) -> Option<&RawLayer> {
        self.layers.get(i_layer.checked_sub(1)?)
    }

    fn raw_layer_mut(&mut self, i_layer: usize) -> Option<&mut RawLayer> {
        self.layers.get_mut(i_layer.checked_sub(1)?)
    }

    unsafe fn raw_layer_unchecked(&self, i_layer: usize) -> &RawLayer {
        unsafe { self.layers.get_unchecked(i_layer.unchecked_sub(1)) }
    }

    unsafe fn raw_layer_unchecked_mut(&mut self, i_layer: usize) -> &mut RawLayer {
        unsafe { self.layers.get_unchecked_mut(i_layer.unchecked_sub(1)) }
    }

    fn params_buffer(&self) -> &[f32] {
        &self.buffer[0..self.za_start]
    }

    fn load_params(&mut self, buffer: &[f32]) -> Result<(), LoadParamsError> {
        if buffer.len() != self.za_start {
            Err(LoadParamsError::IncorrectLength)
        } else {
            unsafe {
                copy_nonoverlapping(buffer.as_ptr(), self.buffer.as_mut_ptr(), self.za_start);
            }
            Ok(())
        }
    }
}

pub struct NeuralNetwork<A: Allocator = Global> {
    n_layers: usize,
    n_inputs: usize,
    n_outputs: usize,
    alloc: A,
    storage: Storage<A>,
}

impl NeuralNetwork<Global> {
    pub fn new<const N_LAYERS: usize>(
        n_inputs: usize,
        layer_descriptions: [LayerDescription; N_LAYERS],
    ) -> Self {
        Self::new_in(Global, n_inputs, layer_descriptions)
    }
}

impl<A: Allocator> NeuralNetwork<A> {
    pub fn new_in<const N_LAYERS: usize>(
        alloc: A,
        n_inputs: usize,
        layer_descriptions: [LayerDescription; N_LAYERS],
    ) -> Self
    where
        A: Clone,
    {
        const {
            assert!(
                N_LAYERS != 0,
                "Neural network must have at least one non-input layer"
            );
        }
        Self {
            n_layers: N_LAYERS,
            n_inputs,
            n_outputs: layer_descriptions.last().unwrap().n_neurons,
            alloc: alloc.clone(),
            storage: Storage::new_in(alloc, n_inputs, layer_descriptions),
        }
    }

    pub fn go_to_gym(&mut self) -> Gym<'_, A>
    where
        A: Clone,
    {
        let mut layer_sizes = Vec::with_capacity_in(self.n_layers(), self.alloc.clone());
        for u in 1..=self.n_layers() {
            layer_sizes.push(self.layer_size(u).unwrap());
        }
        Gym::new(
            self,
            NeuralNetworkDerivs::new_in(self.alloc.clone(), self.n_inputs, layer_sizes),
        )
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn layer_size(&self, i_layer: usize) -> Option<usize> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        Some(raw_layer.n)
    }

    /// Returns `None` if `i_layer` is not in `1..=n_layers`.
    #[allow(unused_variables)]
    pub fn layer_mut<'a, 'b, 'x>(&'a mut self, i_layer: usize) -> Option<LayerMut<'x>>
    where
        'a: 'x,
        'b: 'x,
    {
        let raw_layer = self.storage.raw_layer_mut(i_layer)?;
        Some(unsafe { raw_layer.as_layer_mut() })
    }

    /// # Safety
    ///
    /// `i_layer` must be in `1..n_layers`.
    #[allow(unused_variables)]
    pub unsafe fn layer_unchecked_mut<'a, 'b, 'x>(&'a mut self, i_layer: usize) -> LayerMut<'x>
    where
        'a: 'x,
        'b: 'x,
    {
        unsafe {
            let raw_layer = self.storage.raw_layer_unchecked_mut(i_layer);
            raw_layer.as_layer_mut()
        }
    }

    pub fn randomize(
        &mut self,
        w_range: impl SampleRange<f32> + Clone,
        b_range: impl SampleRange<f32> + Clone,
    ) {
        let mut rng = ThreadRng::default();
        for i_layer in 1..=self.n_layers() {
            let layer = self.layer_mut(i_layer).unwrap();
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

    pub fn forward<'a>(&'a mut self, input_layer: ColRef<f32>) -> ColRef<'a, f32> {
        if input_layer.nrows() != self.n_inputs() {
            let expected = self.n_inputs();
            let found = input_layer.nrows();
            panic!(
                "Neural network is provided incorrect input dimensions (expected {expected}, found {found})"
            );
        }
        for i_layer in 1..=self.n_layers() {
            let a_prev = if i_layer == 1 {
                input_layer
            } else {
                unsafe { self.get_a_unchecked(i_layer - 1).as_col_ref() }
            };
            let mut layer = self.layer_mut(i_layer).unwrap();
            // z = w * a_prev;
            matmul(
                layer.z.rb_mut(),
                Accum::Replace,
                layer.w.rb(),
                a_prev.rb(),
                1.0,
                Par::Seq,
            );
            // z += b; a = phi(z);
            zip!(&mut layer.a, &mut layer.z, &layer.b).for_each(|unzip!(a, z, b)| {
                *z += *b;
                *a = layer.phi.apply(*z);
            });
        }
        self.get_a(self.n_layers()).unwrap()
    }

    /// # Safety
    ///
    /// - `i_layer` must be in range of `1..=n_layers`
    pub unsafe fn get_a_unchecked(&self, i_layer: usize) -> ColPtr<f32> {
        unsafe { self.storage.raw_layer_unchecked(i_layer).a }
    }

    pub fn get_a(&self, i_layer: usize) -> Option<ColRef<'_, f32>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.a.as_col_ref() })
    }

    pub fn get_z(&self, i_layer: usize) -> Option<ColRef<'_, f32>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.z.as_col_ref() })
    }

    pub fn get_b(&self, i_layer: usize) -> Option<ColRef<'_, f32>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.b.as_col_ref() })
    }

    pub fn get_w(&self, i_layer: usize) -> Option<MatRef<'_, f32>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.w.as_mat_ref() })
    }

    /// The continuous buffer that stores all the parameters.
    pub fn params_buffer(&self) -> &[f32] {
        self.storage.params_buffer()
    }

    /// Load all parameters from a buffer of the format produced by `self.params_buffer()`.
    /// `Err` if buffer is of incorrect size.
    pub fn load_params(&mut self, buffer: &[f32]) -> Result<(), LoadParamsError> {
        self.storage.load_params(buffer)
    }
}

pub struct LayerMut<'a> {
    pub n_previous: usize,
    pub n: usize,
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
