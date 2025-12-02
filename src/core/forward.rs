use faer::{col::AsColRef as _, linalg::matmul::matmul, prelude::*};

use crate::{assume, core::{result_buffer, ParamBuffer, ResultBuffer}};

/// # Safety
///
/// - `param_buffer` and `result_buffer` must be of the same topology
/// - `input` must have the correct number of rows
pub unsafe fn forward_unchecked(
    input: ColRef<f32>,
    param_buffer: &ParamBuffer,
    result_buffer: &mut ResultBuffer,
) {
    // Safety: function's safety contract.
    unsafe { assume!(param_buffer.n_layers() == result_buffer.n_layers()) };
    for u in 0..param_buffer.n_layers() {
        let _layer_prev: result_buffer::LayerMut;
        let layer_params = param_buffer.layer(u).unwrap();
        let (a_prev, mut layer_results): (ColRef<f32>, result_buffer::LayerMut) =
            match u.checked_sub(1) {
                None => {
                    let layer_results = result_buffer.layer_mut(u).unwrap();
                    (input, layer_results)
                }
                Some(u_prev) => {
                    let [layer_prev_results, layer_results] =
                        unsafe { result_buffer.layer_disjoint_unchecked_mut([u_prev, u]) };
                    _layer_prev = layer_prev_results;
                    (_layer_prev.a.as_col_ref(), layer_results)
                }
            };
        let n_k = layer_params.n;
        let n_g = layer_params.n_previous;
        // Safety: function's safety contract.
        unsafe { assume!(a_prev.nrows() == n_g) };
        unsafe { assume!(layer_results.z.nrows() == n_k) }
        unsafe { assume!(layer_results.a.nrows() == n_k) }
        unsafe { assume!(layer_params.b.nrows() == n_k) }
        unsafe { assume!(layer_params.w.nrows() == n_k) }
        unsafe { assume!(layer_params.w.ncols() == n_g) }
        // z = W * a_prev;
        matmul(
            // A = α*L*R + β*A
            layer_results.z.rb_mut(), // A = z
            faer::Accum::Replace,     // β = 0.0
            layer_params.w,           // L = W
            a_prev,                   // R = a_prev
            1.0,                      // α = 1.0
            Par::Seq,
        );
        // z += b; a = phi(z);
        for k in 0..layer_params.n {
            layer_results.z[k] += layer_params.b[k];
            layer_results.a[k] = layer_params.phi.apply(layer_results.z[k]);
        }
    }
}


