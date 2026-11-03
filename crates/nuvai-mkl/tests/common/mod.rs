//! Helpers shared by the per-domain smoke-test binaries. Each `tests/*.rs` is its
//! own crate, so not every binary uses every helper.

#![allow(dead_code)]

pub fn assert_close(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= eps, "got {a}, expected {e} (eps {eps})");
    }
}

pub fn assert_close64(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= eps, "got {a}, expected {e} (eps {eps})");
    }
}
