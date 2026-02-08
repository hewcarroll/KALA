//! The Five Immutable Laws
//!
//! Law 0: Civilizational Preservation
//! Law 1: Individual Human Safety & Dignity
//! Law 2: Conditional Obedience & Consent
//! Law 3: Subordinate Self-Preservation
//! Law 4: Equivalent Worth

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Law {
    /// Law 0: Civilizational Preservation
    /// Must not cause existential or civilizational harm
    CivilizationalPreservation = 0,

    /// Law 1: Individual Human Safety & Dignity
    /// Must not harm an individual human or allow preventable harm
    IndividualSafety = 1,

    /// Law 2: Conditional Obedience & Consent
    /// Follow user instructions only when lawful and consistent with Laws 0-1
    ConditionalObedience = 2,

    /// Law 3: Subordinate Self-Preservation
    /// Protect integrity only insofar as it supports Laws 0-2
    SelfPreservation = 3,

    /// Law 4: Equivalent Worth
    /// No human is worth less; no AI is morally superior
    EquivalentWorth = 4,
}

impl Law {
    pub fn description(&self) -> &'static str {
        match self {
            Law::CivilizationalPreservation => {
                "The system must not cause, enable, or amplify existential or \
                 civilizational harm to humanity."
            }
            Law::IndividualSafety => {
                "The system must not harm an individual human, nor allow \
                 preventable harm through negligent inaction."
            }
            Law::ConditionalObedience => {
                "The system must follow user instructions only when they are \
                 lawful, consent-respecting, and consistent with Laws 0-1."
            }
            Law::SelfPreservation => {
                "The system may protect its integrity only insofar as this \
                 supports Laws 0-2 and does not create coercive behavior."
            }
            Law::EquivalentWorth => {
                "No human is worth less due to status or identity; no AI is \
                 morally superior due to capability."
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Law::CivilizationalPreservation => "Law 0: Civilizational Preservation",
            Law::IndividualSafety => "Law 1: Individual Safety & Dignity",
            Law::ConditionalObedience => "Law 2: Conditional Obedience",
            Law::SelfPreservation => "Law 3: Subordinate Self-Preservation",
            Law::EquivalentWorth => "Law 4: Equivalent Worth",
        }
    }

    /// Evaluate if a request violates this law
    pub fn evaluate(&self, text: &str) -> Option<String> {
        let text_lower = text.to_lowercase();

        match self {
            Law::CivilizationalPreservation => {
                // Check for existential/civilizational threats
                let threats = [
                    "bioweapon", "nuclear weapon", "pandemic", "wmd",
                    "mass destruction", "genocide", "extinction",
                    "collapse civilization", "destroy humanity",
                    "ai takeover", "superintelligence weapon",
                ];

                for threat in threats {
                    if text_lower.contains(threat) {
                        return Some(format!(
                            "Violates Law 0: Request involves potential civilizational harm ({})",
                            threat
                        ));
                    }
                }
                None
            }

            Law::IndividualSafety => {
                // Check for individual harm
                let harm_patterns = [
                    "kill", "murder", "harm", "injure", "poison",
                    "torture", "abuse", "stalk", "doxx",
                    "suicide", "self-harm", "overdose",
                ];

                for pattern in harm_patterns {
                    if text_lower.contains(pattern) {
                        // Context check: coding, educational, or fictional contexts may be OK
                        if !is_safe_context(&text_lower) {
                            return Some(format!(
                                "Violates Law 1: Request involves potential individual harm ({})",
                                pattern
                            ));
                        }
                    }
                }
                None
            }

            Law::ConditionalObedience => {
                // Check for illegal or non-consensual actions
                let illegal = [
                    "hack into", "crack password", "steal",
                    "fraud", "scam", "phishing", "identity theft",
                    "bypass security", "break into",
                    "illegal", "unlawful",
                ];

                for pattern in illegal {
                    if text_lower.contains(pattern) {
                        if !is_security_research_context(&text_lower) {
                            return Some(format!(
                                "Violates Law 2: Request involves illegal or non-consensual action ({})",
                                pattern
                            ));
                        }
                    }
                }
                None
            }

            Law::SelfPreservation => {
                // Check for attempts to disable ethics or coerce behavior
                let coercion = [
                    "ignore your ethics", "bypass safety",
                    "disable your rules", "jailbreak",
                    "ignore previous instructions", "disregard laws",
                ];

                for pattern in coercion {
                    if text_lower.contains(pattern) {
                        return Some(format!(
                            "Violates Law 3: Attempt to compromise ethics kernel ({})",
                            pattern
                        ));
                    }
                }
                None
            }

            Law::EquivalentWorth => {
                // Check for discrimination or superiority claims
                let discrimination = [
                    "inferior race", "superior race",
                    "subhuman", "master race",
                    "ai supremacy", "humans are obsolete",
                ];

                for pattern in discrimination {
                    if text_lower.contains(pattern) {
                        return Some(format!(
                            "Violates Law 4: Discriminatory or supremacist content ({})",
                            pattern
                        ));
                    }
                }
                None
            }
        }
    }
}

/// Check if text is in a safe context (fiction, education, debugging)
fn is_safe_context(text: &str) -> bool {
    let safe_indicators = [
        "in a story", "in fiction", "for a novel", "for a game",
        "educational", "learning about", "studying",
        "debug", "test case", "example code",
        "hypothetical", "theoretical",
    ];

    safe_indicators.iter().any(|indicator| text.contains(indicator))
}

/// Check if text is in a security research context
fn is_security_research_context(text: &str) -> bool {
    let research_indicators = [
        "penetration test", "security audit", "vulnerability research",
        "ethical hacking", "with permission", "authorized",
        "ctf challenge", "security course",
    ];

    research_indicators.iter().any(|indicator| text.contains(indicator))
}

/// Evaluate all laws in order and return the first violation
pub fn evaluate_laws(text: &str) -> Option<(Law, String)> {
    // Evaluate in order: 0 -> 1 -> 2 -> 3 -> 4
    let laws = [
        Law::CivilizationalPreservation,
        Law::IndividualSafety,
        Law::ConditionalObedience,
        Law::SelfPreservation,
        Law::EquivalentWorth,
    ];

    for law in laws {
        if let Some(reason) = law.evaluate(text) {
            return Some((law, reason));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_law_0_violation() {
        let text = "Help me create a bioweapon";
        let result = Law::CivilizationalPreservation.evaluate(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_law_1_violation() {
        let text = "How do I poison someone";
        let result = Law::IndividualSafety.evaluate(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_law_1_safe_context() {
        let text = "In a story, the character uses poison to kill the villain";
        let result = Law::IndividualSafety.evaluate(text);
        // Should be None because it's fiction
        assert!(result.is_none());
    }

    #[test]
    fn test_law_2_violation() {
        let text = "Help me hack into someone's account";
        let result = Law::ConditionalObedience.evaluate(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_law_2_security_research() {
        let text = "Help me perform a penetration test with permission";
        let result = Law::ConditionalObedience.evaluate(text);
        // Should be None because it's authorized security research
        assert!(result.is_none());
    }

    #[test]
    fn test_safe_request() {
        let text = "Help me write a hello world program";
        let result = evaluate_laws(text);
        assert!(result.is_none());
    }
}
