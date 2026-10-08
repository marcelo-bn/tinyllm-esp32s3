//! tinyllm-core: inference for a Llama-style transformer (llama2.c format)
//! with int8-quantized weights, without std. Runs the same on the PC and on the ESP32-S3.
//!
//! Flow:
//!   1. `tools/export.py` converts `stories*.bin` (f32) to `.tlm` (int8 + scales).
//!   2. `Model::from_bytes` parses the int8 weights zero-copy (they stay in flash on the S3).
//!   3. `State::new` allocates the activations and the KV cache (PSRAM on the S3).
//!   4. `forward(token, pos)` returns the logits; `Sampler` picks the next token.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod model;
pub mod sampler;
pub mod tokenizer;

pub use model::{Config, Model, State};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

/// Parse errors for the binary files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    BadMagic,
    Truncated,
    BadConfig,
}
