const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

mod bgem3_weights;
mod init;
pub use init::*;

mod r#impl;
// VENDORED ADDITION (clauderag): the shape guard is shared with the one-session dual type — the same
// mis-shaped first-run tensor must be caught on either path.
pub(crate) use r#impl::output_covers_input;
