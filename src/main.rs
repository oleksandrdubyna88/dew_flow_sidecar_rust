//! The dew-flow embedder sidecar.
//!
//! Skeleton entry point: the HTTP surface (`/embed`, `/rerank`, `/health`, `/tokenize`,
//! `/unload`) arrives with the port of the production sidecar. Configuration stays
//! environment-only by design — no config file, no argument parsing; the launcher owns
//! every knob (`PORT`, `ORT_PROVIDER`, `ORT_DEVICE_ID`, `MAX_BATCH`, `EMBED_MAX_LENGTH`, ...).

fn main() {
    println!("dew-flow-sidecar {}", version());
}

/// The version the binary reports; later carried on `/health`.
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::version;

    #[test]
    fn version_matches_the_cargo_manifest() {
        assert_eq!(version(), "0.1.0");
    }
}
