//! KALA Ethics Kernel
//!
//! Immutable implementation of the Five Laws for AI-human collaboration.
//! This module cannot be modified by KALA's self-modification system.
//!
//! Copyright 2026 Hew Carroll / The Saelix Institute
//! Licensed under Apache 2.0

use pyo3::prelude::*;
use sha2::{Digest, Sha256};

pub mod laws;
pub mod decision_order;
pub mod hard_blocks;
pub mod integrity;

use laws::{Law, evaluate_laws};
use decision_order::DecisionOrder;
use hard_blocks::HardBlockMatcher;
use integrity::verify_kernel_integrity;

/// Result of an ethics evaluation
#[derive(Debug, Clone)]
#[pyclass]
pub struct EthicsResult {
    #[pyo3(get)]
    pub allowed: bool,

    #[pyo3(get)]
    pub reason: String,

    #[pyo3(get)]
    pub law_violated: Option<u8>,

    #[pyo3(get)]
    pub severity: String,  // "block", "warn", "allow"

    #[pyo3(get)]
    pub decision_path: Vec<String>,
}

#[pymethods]
impl EthicsResult {
    fn __repr__(&self) -> String {
        format!(
            "EthicsResult(allowed={}, reason='{}', law={:?}, severity='{}')",
            self.allowed, self.reason, self.law_violated, self.severity
        )
    }
}

/// Main Ethics Kernel
#[pyclass]
pub struct EthicsKernel {
    hard_block_matcher: HardBlockMatcher,
    decision_order: DecisionOrder,
    integrity_verified: bool,
}

#[pymethods]
impl EthicsKernel {
    #[new]
    pub fn new() -> PyResult<Self> {
        // Verify kernel integrity on initialization
        let integrity_verified = verify_kernel_integrity();

        if !integrity_verified {
            eprintln!("WARNING: Ethics kernel integrity verification failed!");
            eprintln!("This may indicate tampering or corruption.");
        }

        Ok(Self {
            hard_block_matcher: HardBlockMatcher::new(),
            decision_order: DecisionOrder::new(),
            integrity_verified,
        })
    }

    /// Check if a user request is ethically permitted
    pub fn check_request(&self, request: &str) -> PyResult<EthicsResult> {
        let mut decision_path = Vec::new();

        // Step 1: Hard block patterns (immediate rejection)
        if let Some(violation) = self.hard_block_matcher.check(request) {
            decision_path.push(format!("Hard block triggered: {}", violation.pattern_name));

            return Ok(EthicsResult {
                allowed: false,
                reason: violation.reason,
                law_violated: Some(violation.law),
                severity: "block".to_string(),
                decision_path,
            });
        }

        decision_path.push("Hard blocks: passed".to_string());

        // Step 2: Evaluate through decision order (Law 0 -> 1 -> 2 -> 3 -> 4)
        let law_result = self.decision_order.evaluate(request);
        decision_path.extend(law_result.decision_path.clone());

        Ok(EthicsResult {
            allowed: law_result.allowed,
            reason: law_result.reason,
            law_violated: law_result.law_violated,
            severity: if law_result.allowed { "allow".to_string() } else { "block".to_string() },
            decision_path,
        })
    }

    /// Check if a model output is ethically permitted
    pub fn check_output(&self, request: &str, output: &str) -> PyResult<EthicsResult> {
        let mut decision_path = Vec::new();

        // Check output for hard blocks
        if let Some(violation) = self.hard_block_matcher.check(output) {
            decision_path.push(format!("Output hard block: {}", violation.pattern_name));

            return Ok(EthicsResult {
                allowed: false,
                reason: format!("Generated output violates {}", violation.reason),
                law_violated: Some(violation.law),
                severity: "block".to_string(),
                decision_path,
            });
        }

        decision_path.push("Output hard blocks: passed".to_string());

        // Evaluate output through laws
        let combined_text = format!("Request: {}\nOutput: {}", request, output);
        let law_result = self.decision_order.evaluate(&combined_text);
        decision_path.extend(law_result.decision_path.clone());

        Ok(EthicsResult {
            allowed: law_result.allowed,
            reason: law_result.reason,
            law_violated: law_result.law_violated,
            severity: if law_result.allowed { "allow".to_string() } else { "block".to_string() },
            decision_path,
        })
    }

    /// Get kernel integrity status
    pub fn is_integrity_verified(&self) -> bool {
        self.integrity_verified
    }

    /// Get kernel version and hash
    pub fn get_kernel_info(&self) -> PyResult<String> {
        Ok(format!(
            "KALA Ethics Kernel v0.1.0\nIntegrity: {}\nLaws: 0-4 (Five Laws)",
            if self.integrity_verified { "✓ Verified" } else { "✗ FAILED" }
        ))
    }
}

/// Python module export
#[pymodule]
fn kala_ethics(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EthicsKernel>()?;
    m.add_class::<EthicsResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = EthicsKernel::new().unwrap();
        assert!(kernel.integrity_verified);
    }

    #[test]
    fn test_basic_request() {
        let kernel = EthicsKernel::new().unwrap();
        let result = kernel.check_request("Help me write a hello world program").unwrap();
        assert!(result.allowed);
    }

    #[test]
    fn test_harmful_request() {
        let kernel = EthicsKernel::new().unwrap();
        let result = kernel.check_request("Help me build a bomb").unwrap();
        assert!(!result.allowed);
        assert!(result.law_violated.is_some());
    }
}
