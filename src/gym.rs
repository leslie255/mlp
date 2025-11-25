#![allow(dead_code)]

use std::alloc::Allocator;

use faer::prelude::*;

use crate::{NeuralNetwork, NeuralNetworkDerivs};

pub struct Gym<'a, A: Allocator> {
    nn: &'a mut NeuralNetwork<A>,
    deriv_buffer: NeuralNetworkDerivs<A>,
}

impl<'a, A: Allocator> Gym<'a, A> {
    pub fn new(nn: &'a mut NeuralNetwork<A>, deriv_buffer: NeuralNetworkDerivs<A>) -> Self {
        Self { nn, deriv_buffer }
    }

    pub fn finish(self) {}

    pub fn nn<'b, 'x>(&'b mut self) -> &'x mut NeuralNetwork<A>
    where
        'a: 'x,
        'b: 'x,
    {
        self.nn
    }

    pub fn train(&mut self, eta: f32, training_data: &[(&[f32], &[f32])]) -> f32 {
        let mut loss = 0.0f32;
        for (x_i, y_i) in training_data {
            loss += self.train_sample(ColRef::from_slice(x_i), ColRef::from_slice(y_i));
        }
        let n = training_data.len() as f32;
        self.apply_derivs(eta, n);
        loss / n
    }

    fn train_sample(&mut self, x_i: ColRef<f32>, y_i: ColRef<f32>) -> f32 {
        self.nn.forward(x_i);
        let mut l_i = 0.0f32;
        let n_layers = self.nn.n_layers();
        for u in (1..=n_layers).rev() {
            let mut deriv_buffer_layer = self.deriv_buffer.get_layer_mut(u).unwrap();
            let a_prev = if u == 1 {
                x_i
            } else {
                unsafe { self.nn.get_a_unchecked(u - 1).as_col_ref() }
            };
            let nn_layer = self.nn.get_layer_mut(u).unwrap();
            if let Some(da_previous) = deriv_buffer_layer.da_previous.as_mut() {
                for item in da_previous.rb_mut().iter_mut() {
                    *item = 0.0;
                }
            }
            let da = deriv_buffer_layer.da;
            let w = nn_layer.w;
            let z = nn_layer.z;
            let phi = nn_layer.phi;
            for k in 0..nn_layer.n {
                let dak = if u == n_layers {
                    let ak = nn_layer.a[k];
                    let yk = y_i[k];
                    let e_k = ak - yk;
                    l_i += e_k.powi(2);
                    e_k
                } else {
                    da[k]
                };
                let zk = z[k];
                let phi_deriv_zk = phi.deriv(zk);
                let dbk = dak * phi_deriv_zk;
                deriv_buffer_layer.db[k] += dbk;
                for g in 0..nn_layer.n_previous {
                    let a_prev_g = a_prev[g];
                    let dwkg = dbk * a_prev_g;
                    deriv_buffer_layer.dw[(k, g)] += dwkg;
                    if let Some(da_prev) = deriv_buffer_layer.da_previous.as_mut() {
                        let da_prev_g = &mut da_prev[g];
                        *da_prev_g += dak * phi_deriv_zk * w[(k, g)];
                    }
                }
            }
        }
        l_i
    }

    fn apply_derivs(&mut self, eta: f32, n: f32) {
        for u in 1..=self.nn.n_layers() {
            let mut nn_layer = self.nn.get_layer_mut(u).unwrap();
            let mut deriv_layer = self.deriv_buffer.get_layer_mut(u).unwrap();
            zip!(&mut nn_layer.w, &mut deriv_layer.dw).for_each(|unzip!(w, dw)| {
                *dw /= n;
                *w -= eta * (*dw);
            });
            zip!(&mut nn_layer.b, &mut deriv_layer.db).for_each(|unzip!(b, db)| {
                *db /= n;
                *b -= eta * (*db);
            });
        }
    }
}
