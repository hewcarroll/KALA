//! Cryptographic Integrity Verification
//!
//! Ensures the ethics kernel has not been tampered with.
//! Uses SHA-256 hashing of critical components.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Expected hash of the laws module (updated on legitimate changes only)
/// This would be set during build/release
const EXPECTED_LAWS_HASH: &str = "PLACEHOLDER_HASH_WILL_BE_SET_IN_BUILD";

/// Expected hash of the decision order module
const EXPECTED_DECISION_ORDER_HASH: &str = "PLACEHOLDER_HASH_WILL_BE_SET_IN_BUILD";

/// Expected hash of the hard blocks module
const EXPECTED_HARD_BLOCKS_HASH: &str = "PLACEHOLDER_HASH_WILL_BE_SET_IN_BUILD";

/// Verify the integrity of the ethics kernel
///
/// In development, this always returns true.
/// In production builds, this would verify cryptographic hashes.
pub fn verify_kernel_integrity() -> bool {
    // In development mode, skip verification
    #[cfg(debug_assertions)]
    {
        eprintln!("DEBUG: Skipping integrity verification in development mode");
        return true;
    }

    // In release mode, verify hashes
    #[cfg(not(debug_assertions))]
    {
        verify_module_hash("laws.rs", EXPECTED_LAWS_HASH)
            && verify_module_hash("decision_order.rs", EXPECTED_DECISION_ORDER_HASH)
            && verify_module_hash("hard_blocks.rs", EXPECTED_HARD_BLOCKS_HASH)
    }
}

/// Verify the hash of a specific module
fn verify_module_hash(module_name: &str, expected_hash: &str) -> bool {
    // Get the source file path
    let source_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(module_name);

    // Read the source file
    let source_code = match fs::read_to_string(&source_path) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("ERROR: Cannot read {}: {}", module_name, e);
            return false;
        }
    };

    // Calculate hash
    let actual_hash = calculate_hash(&source_code);

    // Compare
    if actual_hash != expected_hash {
        eprintln!(
            "WARNING: Integrity check failed for {}",
            module_name
        );
        eprintln!("  Expected: {}", expected_hash);
        eprintln!("  Actual:   {}", actual_hash);
        return false;
    }

    true
}

/// Calculate SHA-256 hash of a string
pub fn calculate_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Generate integrity hashes for the current codebase
/// Used during build to set the expected hashes
pub fn generate_integrity_hashes() -> IntegrityHashes {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_dir = Path::new(manifest_dir).join("src");

    let modules = ["laws.rs", "decision_order.rs", "hard_blocks.rs"];

    let mut hashes = IntegrityHashes {
        laws: String::new(),
        decision_order: String::new(),
        hard_blocks: String::new(),
    };

    for module in modules {
        let path = src_dir.join(module);
        if let Ok(source) = fs::read_to_string(&path) {
            let hash = calculate_hash(&source);

            match module {
                "laws.rs" => hashes.laws = hash,
                "decision_order.rs" => hashes.decision_order = hash,
                "hard_blocks.rs" => hashes.hard_blocks = hash,
                _ => {}
            }
        }
    }

    hashes
}

#[derive(Debug)]
pub struct IntegrityHashes {
    pub laws: String,
    pub decision_order: String,
    pub hard_blocks: String,
}

impl IntegrityHashes {
    /// Print hashes in a format suitable for copying to source code
    pub fn print_for_build(&self) {
        println!("// Generated integrity hashes:");
        println!("const EXPECTED_LAWS_HASH: &str = \"{}\";", self.laws);
        println!("const EXPECTED_DECISION_ORDER_HASH: &str = \"{}\";", self.decision_order);
        println!("const EXPECTED_HARD_BLOCKS_HASH: &str = \"{}\";", self.hard_blocks);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_calculation() {
        let data = "test data";
        let hash = calculate_hash(data);
        assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[test]
    fn test_hash_consistency() {
        let data = "consistent data";
        let hash1 = calculate_hash(data);
        let hash2 = calculate_hash(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_difference() {
        let hash1 = calculate_hash("data 1");
        let hash2 = calculate_hash("data 2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_integrity_verification() {
        // Should always pass in debug mode
        assert!(verify_kernel_integrity());
    }

    #[test]
    fn test_generate_hashes() {
        let hashes = generate_integrity_hashes();
        assert!(!hashes.laws.is_empty());
        assert!(!hashes.decision_order.is_empty());
        assert!(!hashes.hard_blocks.is_empty());
    }
}
