use std::fmt::{self, Debug, Display};

use faer::prelude::*;

use crate::{deriv_buffer, param_buffer};

pub struct PrettyPrintParams<'a> {
    i_layer: usize,
    layer: param_buffer::LayerRef<'a>,
}

impl<'a> PrettyPrintParams<'a> {
    pub fn new(i_layer: usize, layer: param_buffer::LayerRef<'a>) -> Self {
        Self { i_layer, layer }
    }
}

impl Debug for PrettyPrintParams<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

fn n_digits(u: usize) -> usize {
    match u {
        0 => 1,
        u => ((u as f32).log10() + 1.0) as usize,
    }
}

impl Display for PrettyPrintParams<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let w = self.layer.w.rb();
        let b = self.layer.b.rb();
        let center_line = self.layer.n / 2;
        let phi = self.layer.phi.name();
        let i_layer = self.i_layer;
        let i_layer_length = n_digits(i_layer);
        let i_previous_layer_length = match i_layer.checked_sub(1) {
            None => 1, // "x"
            Some(i_previous) => n_digits(i_previous),
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
                match i_layer.checked_sub(1) {
                    None => write!(f, " x + ")?,
                    Some(i_previous) => write!(f, " a_{} + ", i_previous)?,
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

pub struct PrettyPrintDerivs<'a> {
    i_layer: usize,
    layer: deriv_buffer::LayerRef<'a>,
}

impl<'a> PrettyPrintDerivs<'a> {
    pub fn new(i_layer: usize, layer: deriv_buffer::LayerRef<'a>) -> Self {
        Self { i_layer, layer }
    }
}

impl Debug for PrettyPrintDerivs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for PrettyPrintDerivs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dw = self.layer.dw.rb();
        let db = self.layer.db.rb();
        let center_line = self.layer.n / 2;
        let i_layer = self.i_layer;
        let i_layer_length = n_digits(i_layer);
        let i_previous_layer_length = match i_layer.checked_sub(1) {
            None => 1, // "x"
            Some(i_previous) => n_digits(i_previous),
        };
        for i_line in 0..self.layer.n {
            if i_line == center_line {
                write!(f, "a_{i_layer} = phi(")?;
            } else {
                write!(f, "      ")?;
                for _ in 0..(3 + i_layer_length) {
                    write!(f, " ")?;
                }
            }
            write!(f, "[")?;
            let mut iter = dw.row(i_line).iter();
            while let Some(&element) = iter.next() {
                if element.is_sign_positive() {
                    write!(f, " {:.012?}", element)?;
                } else {
                    write!(f, "{:.012?}", element)?;
                }
                if iter.size_hint().0 != 0 {
                    write!(f, " ")?;
                }
            }
            write!(f, "]")?;
            if i_line == center_line {
                match i_layer.checked_sub(1) {
                    None => write!(f, " x + ")?,
                    Some(i_previous) => write!(f, " a_{} + ", i_previous)?,
                }
            } else {
                write!(f, "    ")?;
                for _ in 0..i_previous_layer_length {
                    write!(f, " ")?;
                }
            }
            write!(f, "[")?;
            let b_element = db.get(i_line);
            if b_element.is_sign_positive() {
                write!(f, " {:.012?}", b_element)?;
            } else {
                write!(f, "{:.012?}", b_element)?;
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
