/// `assert_unchecked` in release, `assert` in debug.
#[macro_export]
macro_rules! assume {
    ($predicate:expr) => {{
        if cfg!(debug_assertions) {
            assert!($predicate);
        } else {
            std::hint::assert_unchecked($predicate);
        }
    }};
}


