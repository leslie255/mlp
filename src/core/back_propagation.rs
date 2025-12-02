use std::iter;

use faer::prelude::*;

use crate::{
    assume,
    core::{deriv_buffer, forward_unchecked, param_buffer, result_buffer, DerivBuffer, ParamBuffer, ResultBuffer},
};

/// Calculates and applies derivative.
///
/// Returns loss over the provided samples.
///
/// # Safety
///
/// - `param_buffer`, `result_buffer` and `deriv_buffer` must be of the same topology
/// - all inputs and outputs in `samples` must be of the correct sizes
pub unsafe fn calculate_derivs<'a>(
    param_buffer: &ParamBuffer,
    result_buffer: &mut ResultBuffer,
    deriv_buffer: &mut DerivBuffer,
    samples: impl IntoIterator<Item = &'a (&'a [f32], &'a [f32])>,
) -> f32 {
    unsafe { assume!(param_buffer.n_layers() == result_buffer.n_layers()) };
    unsafe { assume!(result_buffer.n_layers() == deriv_buffer.n_layers()) };
    let mut loss = 0.0f32;
    deriv_buffer.clear_params();
    let mut n = 0usize;
    for (x_i, y_i) in samples {
        n += 1;
        loss += unsafe {
            back_propagate_sample(
                param_buffer,
                result_buffer,
                deriv_buffer,
                ColRef::from_slice(x_i),
                ColRef::from_slice(y_i),
            )
        };
    }
    let n = n as f32;
    for p in deriv_buffer.params_mut() {
        *p /= n;
    }
    loss / n
}

/// Calculates and applies derivative.
///
/// Returns loss over the provided samples.
///
/// # Safety
///
/// - `param_buffer`, `result_buffer` and `deriv_buffer` must be of the same topology
/// - all inputs and outputs in `samples` must be of the correct sizes
pub unsafe fn apply_derivs(param_buffer: &mut ParamBuffer, deriv_buffer: &DerivBuffer, eta: f32) {
    // Params buffer and deriv buffer has the same layout for the weights and biases (deriv buffer
    // has an additional da section at the end, but it does not affect the layout for its param
    // section).
    let param_buffer = param_buffer.as_mut_slice();
    let deriv_param_buffer = deriv_buffer.params();
    unsafe { assume!(param_buffer.len() == deriv_param_buffer.len()) };
    for (p, dp) in iter::zip(param_buffer, deriv_param_buffer) {
        *p -= eta * (*dp);
    }
}

#[inline(always)]
unsafe fn back_propagate_sample(
    param_buffer: &ParamBuffer,
    result_buffer: &mut ResultBuffer,
    deriv_buffer: &mut DerivBuffer,
    x: ColRef<f32>,
    y: ColRef<f32>,
) -> f32 {
    unsafe { forward_unchecked(x, param_buffer, result_buffer) };
    let mut l_i = 0.0f32;
    let n_layers = param_buffer.n_layers();
    for u in (0..n_layers).rev() {
        let u_prev = u.checked_sub(1);
        let a_prev = match u_prev {
            None => x,
            Some(u_prev) => unsafe { result_buffer.layer_unchecked(u_prev).a },
        };
        let layer_results = result_buffer.layer(u).unwrap();
        let (da_prev, layer_derivs) = match u_prev {
            None => (None, deriv_buffer.layer_mut(u).unwrap()),
            Some(u_prev) => {
                let [deriv_layer_prev, deriv_layer] =
                    unsafe { deriv_buffer.layer_disjoint_unchecked_mut([u_prev, u]) };
                (Some(deriv_layer_prev.da), deriv_layer)
            }
        };
        let nn_layer = param_buffer.layer(u).unwrap();
        let is_output_layer = u + 1 == n_layers;
        let n_k = layer_results.n;
        let n_g = layer_results.n_previous;
        unsafe { assume!(a_prev.nrows() == n_g) };
        unsafe { assume!(layer_results.z.nrows() == n_k) }
        unsafe { assume!(layer_results.a.nrows() == n_k) }
        if is_output_layer {
            unsafe { assume!(layer_results.a.nrows() == layer_results.n) };
            unsafe { assume!(y.nrows() == layer_results.n) };
            for k in 0..layer_results.n {
                l_i += (layer_results.a[k] - y[k]).powi(2);
            }
        }
        unsafe {
            back_propagate_layer(
                is_output_layer,
                a_prev,
                nn_layer,
                layer_derivs,
                layer_results,
                da_prev,
                y,
            );
        }
    }
    l_i
}

#[inline(always)]
unsafe fn back_propagate_layer(
    is_output_layer: bool,
    a_prev: ColRef<f32>,
    layer_params: param_buffer::LayerRef,
    mut layer_derivs: deriv_buffer::LayerMut,
    layer_results: result_buffer::LayerRef,
    mut da_prev: Option<ColMut<f32>>,
    y: ColRef<f32>,
) {
    let n_k = layer_params.n;
    let n_g = layer_params.n_previous;
    let phi = layer_params.phi;
    let w = layer_params.w;
    let a = layer_results.a;
    let z = layer_results.z;
    let da = layer_derivs.da;
    unsafe { assume!(w.nrows() == n_k) };
    unsafe { assume!(w.ncols() == n_g) };
    unsafe { assume!(a.nrows() == n_k) };
    unsafe { assume!(a_prev.nrows() == n_g) };
    unsafe { assume!(z.nrows() == n_k) };
    unsafe { assume!(a.nrows() == n_k) };
    unsafe { assume!(da.nrows() == n_k) };
    // Zero da_prev for the summing that happens later.
    // `da` is a per-sample vector, which is unlike `dw` and `db`.
    if let Some(ref mut da_prev) = da_prev {
        for dag in da_prev.rb_mut().iter_mut() {
            *dag = 0.0;
        }
    }
    for k in 0..n_k {
        let phi_deriv_z = phi.deriv(z[k]);
        let dak = match is_output_layer {
            // da[k] = e[k] for output layer.
            true => a[k] - y[k],
            // Next layer have calculated it for us. (We're iterating through layers backwards)
            false => da[k],
        };
        layer_derivs.db[k] += dak * phi_deriv_z;
        for g in 0..n_g {
            layer_derivs.dw[(k, g)] += dak * phi_deriv_z * a_prev[g];
            // Calculate da for the previous layer.
            if let Some(ref mut da_prev) = da_prev {
                da_prev[g] += dak * phi_deriv_z * w[(k, g)];
            }
        }
    }
}
