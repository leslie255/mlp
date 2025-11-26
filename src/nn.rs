use std::{
    alloc::{Allocator, Global},
    fmt::Debug,
    iter,
    ops::{AddAssign, Mul},
    ptr::{NonNull, copy_nonoverlapping, write_bytes},
};

use faer::prelude::*;

use rand::{
    Rng,
    distr::uniform::{SampleRange, SampleUniform},
    rngs::ThreadRng,
};

use derive_more::{Display, Error};

use crate::{
    ActivationFunction, ActivationFunctionVTable, ColPtr, Gym, MatPtr, NeuralNetworkDerivs,
    PrettyPrintParams,
};

#[derive(Debug)]
pub struct LayerDescription<T> {
    pub n_neurons: usize,
    pub phi: ActivationFunctionVTable<T>,
}

impl<T> LayerDescription<T> {
    pub fn new(n_neurons: usize, phi: impl ActivationFunction<T> + 'static) -> Self {
        Self {
            n_neurons,
            phi: ActivationFunctionVTable::new(phi),
        }
    }
}

struct RawLayer<T> {
    n_previous: usize,
    n: usize,
    w: MatPtr<T>,
    b: ColPtr<T>,
    phi: ActivationFunctionVTable<T>,
    z: ColPtr<T>,
    a: ColPtr<T>,
}

impl<T> RawLayer<T> {
    unsafe fn as_layer_mut(&mut self) -> LayerMut<'_, T> {
        unsafe {
            LayerMut {
                n_previous: self.n_previous,
                n: self.n,
                phi: self.phi,
                w: self.w.as_mat_mut(),
                b: self.b.as_col_mut(),
                z: self.z.as_col_mut(),
                a: self.a.as_col_mut(),
            }
        }
    }
}

struct Storage<T, A: Allocator> {
    layers: Box<[RawLayer<T>], A>,
    buffer: Box<[T], A>,
    /// Start of the region in `buffer` where storage of activation results (z and a) begin.
    za_start: usize,
}

#[derive(Debug, Clone, Display, Error)]
pub enum LoadParamsError {
    #[display("incorrect length")]
    IncorrectLength,
}

impl<T, A: Allocator> Storage<T, A> {
    fn new_in<const N_LAYERS: usize>(
        alloc: A,
        n_inputs: usize,
        layer_descriptions: [LayerDescription<T>; N_LAYERS],
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
        let buffer: Box<[T], A> = {
            let mut buffer = Box::new_uninit_slice_in(n_floats, alloc.clone());
            bytemuck::fill_zeroes(&mut buffer);
            unsafe {
                write_bytes(buffer.as_mut_ptr(), 0u8, n_floats);
                buffer.assume_init()
            }
        };
        let layers: Box<[RawLayer<T>], A> = {
            let mut layers = Box::new_uninit_slice_in(n_layers, alloc);
            let data: NonNull<T> = NonNull::from(buffer.as_ref()).cast();
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

    fn raw_layer(&self, i_layer: usize) -> Option<&RawLayer<T>> {
        self.layers.get(i_layer.checked_sub(1)?)
    }

    fn raw_layer_mut(&mut self, i_layer: usize) -> Option<&mut RawLayer<T>> {
        self.layers.get_mut(i_layer.checked_sub(1)?)
    }

    unsafe fn raw_layer_unchecked(&self, i_layer: usize) -> &RawLayer<T> {
        unsafe { self.layers.get_unchecked(i_layer.unchecked_sub(1)) }
    }

    unsafe fn raw_layer_unchecked_mut(&mut self, i_layer: usize) -> &mut RawLayer<T> {
        unsafe { self.layers.get_unchecked_mut(i_layer.unchecked_sub(1)) }
    }

    fn params_buffer(&self) -> &[T] {
        &self.buffer[0..self.za_start]
    }

    fn load_params(&mut self, buffer: &[T]) -> Result<(), LoadParamsError> {
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

pub struct NeuralNetwork<T, A: Allocator = Global> {
    n_layers: usize,
    n_inputs: usize,
    n_outputs: usize,
    alloc: A,
    storage: Storage<T, A>,
}

impl<T> NeuralNetwork<T, Global> {
    pub fn new<const N_LAYERS: usize>(
        n_inputs: usize,
        layer_descriptions: [LayerDescription<T>; N_LAYERS],
    ) -> Self {
        Self::new_in(n_inputs, layer_descriptions, Global)
    }
}

impl<T, A: Allocator> NeuralNetwork<T, A> {
    pub fn new_in<const N_LAYERS: usize>(
        n_inputs: usize,
        layer_descriptions: [LayerDescription<T>; N_LAYERS],
        alloc: A,
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
    pub fn layer_mut<'a, 'b, 'x>(&'a mut self, i_layer: usize) -> Option<LayerMut<'x, T>>
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
    pub unsafe fn layer_unchecked_mut<'a, 'b, 'x>(&'a mut self, i_layer: usize) -> LayerMut<'x, T>
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
        w_range: impl SampleRange<T> + Clone,
        b_range: impl SampleRange<T> + Clone,
    ) where
        T: SampleUniform,
    {
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

    pub fn forward<'a>(&'a mut self, input_layer: &[T]) -> ColRef<'a, T>
    where
        T: AddAssign + Copy + Mul<T, Output = T> + 'static,
    {
        if input_layer.len() != self.n_inputs() {
            let expected = self.n_inputs();
            let found = input_layer.len();
            panic!(
                "Neural network is provided incorrect input dimensions (expected {expected}, found {found})"
            );
        }
        for i_layer in 1..=self.n_layers() {
            let a_prev = if i_layer == 1 {
                input_layer.as_ptr()
            } else {
                unsafe { self.get_a_unchecked(i_layer - 1).ptr.as_ptr() }
            };
            let layer = unsafe { self.storage.raw_layer_unchecked_mut(i_layer) };
            Self::forward_layer(
                layer.n,
                layer.n_previous,
                layer.a.ptr.as_ptr(),
                layer.z.ptr.as_ptr(),
                layer.w.ptr.as_ptr(),
                layer.b.ptr.as_ptr(),
                a_prev,
                layer.phi.apply,
            );
            // // z = W * a + b;
            // // a = phi(z);
            // for k in 0..layer.n {
            //     // z[k] = w[k][g] * a_prev[k];
            //     let zk = unsafe { layer.z.rb_mut().get_mut_unchecked(k) };
            //     let ak = unsafe { layer.a.rb_mut().get_mut_unchecked(k) };
            //     let bk = unsafe { layer.b.rb().get_unchecked(k) };
            //     for g in 0..layer.n_previous {
            //         let wkg = unsafe { *layer.w.rb().get_unchecked(k, g) };
            //         let a_prev_g = unsafe { *a_prev.rb().get_unchecked(g) };
            //         *zk = wkg * a_prev_g;
            //     }
            //     // z[k] += b[k];
            //     *zk += *bk;
            //     // a[k] = phi(z[k]);
            //     *ak = layer.phi.apply(*zk);
            // }
        }
        self.get_a(self.n_layers()).unwrap()
    }

    fn forward_layer(
        n: usize,
        n_prev: usize,
        a: *mut T,
        z: *mut T,
        w: *const T,
        b: *const T,
        a_prev: *const T,
        phi: fn(T) -> T,
    ) where
        T: AddAssign + Copy + Mul<T, Output = T> + 'static,
    {
        // z = W * a + b;
        // a = phi(z);
        unsafe {
            for k in 0..n {
                // z[k] = w[k][g] * a_prev[k];
                let zk = z.add(k);
                let ak = a.add(k);
                let bk = b.add(k);
                for g in 0..n_prev {
                    let wkg = *w.add(g * n_prev + k);
                    let a_prev_g = *a_prev.add(g);
                    *zk = wkg * a_prev_g;
                }
                // z[k] += b[k];
                *zk += *bk;
                // a[k] = phi(z[k]);
                *ak = phi(*zk);
            }
        }
    }

    /// # Safety
    ///
    /// - `i_layer` must be in range of `1..=n_layers`
    pub unsafe fn get_a_unchecked(&self, i_layer: usize) -> ColPtr<T> {
        unsafe { self.storage.raw_layer_unchecked(i_layer).a }
    }

    pub fn get_a(&self, i_layer: usize) -> Option<ColRef<'_, T>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.a.as_col_ref() })
    }

    pub fn get_z(&self, i_layer: usize) -> Option<ColRef<'_, T>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.z.as_col_ref() })
    }

    pub fn get_b(&self, i_layer: usize) -> Option<ColRef<'_, T>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.b.as_col_ref() })
    }

    pub fn get_w(&self, i_layer: usize) -> Option<MatRef<'_, T>> {
        let raw_layer = self.storage.raw_layer(i_layer)?;
        // Safety: `self` is under `&` reference.
        Some(unsafe { raw_layer.w.as_mat_ref() })
    }

    /// The continuous buffer that stores all the parameters.
    pub fn params_buffer(&self) -> &[T] {
        self.storage.params_buffer()
    }

    /// Load all parameters from a buffer of the format produced by `self.params_buffer()`.
    /// `Err` if buffer is of incorrect size.
    pub fn load_params(&mut self, buffer: &[T]) -> Result<(), LoadParamsError> {
        self.storage.load_params(buffer)
    }
}

#[derive(Debug)]
pub struct LayerMut<'a, T> {
    pub n_previous: usize,
    pub n: usize,
    pub phi: ActivationFunctionVTable<T>,
    pub w: MatMut<'a, T>,
    pub b: ColMut<'a, T>,
    pub z: ColMut<'a, T>,
    pub a: ColMut<'a, T>,
}

impl<A: Allocator> NeuralNetwork<f32, A> {
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
}

impl<'a, T> LayerMut<'a, T>
where
    T: Debug,
{
    pub fn pretty_print_params(&'a self, i_layer: usize) -> PrettyPrintParams<'a, T> {
        PrettyPrintParams::new(i_layer, self)
    }
}
