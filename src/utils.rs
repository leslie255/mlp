use std::{
    alloc::{Allocator, Layout},
    slice,
};

use bytemuck::Zeroable;

pub(crate) fn zeroed_box_slice_in<A: Allocator, T: Zeroable>(
    length: usize,
    alloc: A,
) -> Box<[T], A> {
    let layout = Layout::array::<T>(length).unwrap();
    let ptr: *mut T = alloc.allocate_zeroed(layout).unwrap().cast().as_ptr();
    let ptr: *mut [T] = unsafe { slice::from_raw_parts_mut(ptr, length) };
    unsafe { Box::from_raw_in(ptr, alloc) }
}
