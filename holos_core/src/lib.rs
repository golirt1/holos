//! # HOLOS — a fast, dependency-free Hyperdimensional Computing (HDC / VSA) engine.
//!
//! HDC represents information as very high-dimensional vectors (~10,000-D) and computes
//! with three cheap algebraic operations plus an associative memory:
//!
//! - **bind** (`⊗`): associate two hypervectors (XOR) — invertible, result orthogonal to both.
//! - **bundle** (`⊕`): superpose several (majority) — result similar to all of them.
//! - **permute** (`ρ`): rotate — encode order / sequences.
//! - **cleanup**: nearest-neighbor search over an [`ItemMemory`] to decode a noisy result.
//!
//! The flagship [`Hypervector`] uses the **BSC** (Binary Spatter Code) model, bit-packed into
//! `u64` words so operations become XOR + hardware popcount — very fast and cache-friendly.
//! A bipolar [`map::MapVector`] model is also provided for tasks that favor linear bundling.
//!
//! ```
//! use holos_core::{Hypervector, Rng};
//! let mut rng = Rng::new(42);
//! let d = 10_000;
//! let role = Hypervector::random(d, &mut rng);
//! let value = Hypervector::random(d, &mut rng);
//! let bound = role.bind(&value);                 // associate role↔value
//! assert!(bound.bind(&role).similarity(&value) > 0.99); // unbind recovers value
//! ```

#![forbid(unsafe_code)]

mod bsc;
pub mod map;
mod memory;
mod rng;

pub use bsc::{bundle, permute, Hypervector};
pub use memory::ItemMemory;
pub use rng::Rng;
