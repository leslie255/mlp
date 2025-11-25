use std::fmt::{self, Debug, Display};

use faer::prelude::*;

use crate::LayerMut;

pub struct PrettyPrintParams<'a> {
    i_layer: usize,
    layer: &'a LayerMut<'a>,
}

impl<'a> PrettyPrintParams<'a> {
    pub fn new(i_layer: usize, layer: &'a LayerMut<'a>) -> Self {
        Self { i_layer, layer }
    }
}

impl Debug for PrettyPrintParams<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for PrettyPrintParams<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let w = self.layer.w.rb();
        let b = self.layer.b.rb();
        let center_line = self.layer.n / 2;
        let phi = self.layer.phi.name();
        let i_layer = self.i_layer;
        let i_layer_length = ((i_layer as f32).log10() + 1.0) as usize;
        let i_previous_layer_length = match i_layer {
            1 => 1, // "x"
            i_layer => (((i_layer - 1) as f32).log10() + 1.0) as usize + 2,
        };
        for i_line in 0..self.layer.n {
            if i_line == center_line {
                write!(f, "a_{i_layer} = {phi}(")?;
            } else {
                write!(f, "      ")?;
                for _ in 0..(phi.len() + i_layer_length) {
                    write!(f, " ")?;
                }
            }
            write!(f, "[")?;
            let mut iter = w.row(i_line).iter();
            while let Some(&element) = iter.next() {
                if element.is_sign_positive() {
                    write!(f, " {:.04?}", element)?;
                } else {
                    write!(f, "{:.04?}", element)?;
                }
                if iter.size_hint().0 != 0 {
                    write!(f, " ")?;
                }
            }
            write!(f, "]")?;
            if i_line == center_line {
                match i_layer {
                    1 => write!(f, " x + ")?,
                    i_layer => write!(f, " a_{} + ", i_layer - 1)?,
                }
            } else {
                write!(f, "    ")?;
                for _ in 0..i_previous_layer_length {
                    write!(f, " ")?;
                }
            }
            write!(f, "[")?;
            let b_element = b.get(i_line);
            if b_element.is_sign_positive() {
                write!(f, " {:.04?}", b_element)?;
            } else {
                write!(f, "{:.04?}", b_element)?;
            }
            if iter.size_hint().0 != 0 {
                write!(f, " ")?;
            }
            write!(f, "]")?;
            if i_line == center_line {
                write!(f, ")")?;
            }
            if i_line != self.layer.n - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
