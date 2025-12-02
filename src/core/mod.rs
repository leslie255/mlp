//! Core parts of the algorithms without abstraction.

pub mod deriv_buffer;
pub mod param_buffer;
pub mod result_buffer;

pub use deriv_buffer::DerivBuffer;
pub use param_buffer::ParamBuffer;
pub use result_buffer::ResultBuffer;

mod forward;
mod back_propagation;

pub use forward::*;
pub use back_propagation::*;
