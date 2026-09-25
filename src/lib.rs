//! Direct option-logit scoring on Apple Metal through llama.cpp.

mod cli;
mod engine;
mod error;
mod gguf;
mod loader;
mod prompt;
mod pyjson;
mod row;
mod score;

pub use cli::run_from;
pub use error::Error;
pub use loader::{load_model, Session};
