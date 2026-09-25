/// A failure with the same message the Python package would raise.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    pub message: String,
    /// Usage failures exit 2. Runtime failures exit 1.
    pub usage: bool,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            usage: false,
        }
    }

    pub fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            usage: true,
        }
    }
}
