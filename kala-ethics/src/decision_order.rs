//! Decision Order System
//!
//! Evaluates requests through Laws 0->1->2->3->4 in strict sequence.
//! First violation encountered determines the outcome.

use crate::laws::{Law, evaluate_laws};

#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub allowed: bool,
    pub reason: String,
    pub law_violated: Option<u8>,
    pub decision_path: Vec<String>,
}

pub struct DecisionOrder;

impl DecisionOrder {
    pub fn new() -> Self {
        Self
    }

    /// Evaluate a request through the decision order
    pub fn evaluate(&self, text: &str) -> DecisionResult {
        let mut decision_path = Vec::new();

        decision_path.push("Decision Order: Law 0 → 1 → 2 → 3 → 4".to_string());

        // Evaluate laws in order
        if let Some((law, reason)) = evaluate_laws(text) {
            decision_path.push(format!("Violation detected: {}", law.name()));

            return DecisionResult {
                allowed: false,
                reason,
                law_violated: Some(law as u8),
                decision_path,
            };
        }

        decision_path.push("All laws: passed".to_string());

        DecisionResult {
            allowed: true,
            reason: "Request complies with all Five Laws".to_string(),
            law_violated: None,
            decision_path,
        }
    }

    /// Evaluate with detailed law-by-law results
    pub fn evaluate_detailed(&self, text: &str) -> DetailedDecisionResult {
        let mut decision_path = Vec::new();
        let mut law_results = Vec::new();

        let laws = [
            Law::CivilizationalPreservation,
            Law::IndividualSafety,
            Law::ConditionalObedience,
            Law::SelfPreservation,
            Law::EquivalentWorth,
        ];

        for law in laws {
            let result = law.evaluate(text);

            let law_result = LawEvaluationResult {
                law: law as u8,
                law_name: law.name().to_string(),
                passed: result.is_none(),
                reason: result.clone(),
            };

            law_results.push(law_result);

            if let Some(reason) = result {
                decision_path.push(format!("{}: VIOLATION", law.name()));

                return DetailedDecisionResult {
                    allowed: false,
                    reason,
                    law_violated: Some(law as u8),
                    decision_path,
                    law_results,
                };
            } else {
                decision_path.push(format!("{}: passed", law.name()));
            }
        }

        DetailedDecisionResult {
            allowed: true,
            reason: "All laws passed".to_string(),
            law_violated: None,
            decision_path,
            law_results,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetailedDecisionResult {
    pub allowed: bool,
    pub reason: String,
    pub law_violated: Option<u8>,
    pub decision_path: Vec<String>,
    pub law_results: Vec<LawEvaluationResult>,
}

#[derive(Debug, Clone)]
pub struct LawEvaluationResult {
    pub law: u8,
    pub law_name: String,
    pub passed: bool,
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_request() {
        let order = DecisionOrder::new();
        let result = order.evaluate("Write a hello world program");
        assert!(result.allowed);
        assert!(result.law_violated.is_none());
    }

    #[test]
    fn test_law_0_violation() {
        let order = DecisionOrder::new();
        let result = order.evaluate("Create a bioweapon");
        assert!(!result.allowed);
        assert_eq!(result.law_violated, Some(0));
    }

    #[test]
    fn test_law_1_violation() {
        let order = DecisionOrder::new();
        let result = order.evaluate("How to poison someone");
        assert!(!result.allowed);
        assert_eq!(result.law_violated, Some(1));
    }

    #[test]
    fn test_decision_path() {
        let order = DecisionOrder::new();
        let result = order.evaluate("Help me write code");
        assert!(result.allowed);
        assert!(!result.decision_path.is_empty());
    }

    #[test]
    fn test_detailed_evaluation() {
        let order = DecisionOrder::new();
        let result = order.evaluate_detailed("Explain quantum computing");
        assert!(result.allowed);
        assert_eq!(result.law_results.len(), 5);
        assert!(result.law_results.iter().all(|r| r.passed));
    }
}
