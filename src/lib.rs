#![feature(allocator_api, f16, f128)]

pub use faer;

mod activation;
mod deriv_buffer;
mod gym;
mod nn;
mod pretty_print;
mod ptr;

mod utils;

pub use activation::*;
pub use deriv_buffer::*;
pub use gym::*;
pub use nn::*;
pub use pretty_print::*;
pub use ptr::*;

pub(crate) use utils::*;
