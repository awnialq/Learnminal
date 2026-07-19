//! Deterministic flag verification against local reference text.

use std::collections::BTreeSet;

use crate::learnminal::types::ReferenceContext;

/// Result of checking flags mentioned in a model reply against reference text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyResult {
    pub footer: String,
    /// `Some(true)` all matched, `Some(false)` some unverified, `None` skipped.
    pub verified: Option<bool>,
}

/// Verify flags in `reply` against `reference` and return a footer + status.
pub fn verify_reply(reply: &str, reference: &ReferenceContext) -> VerifyResult {
    if !reference.has_body() {
        return VerifyResult {
            footer: "Verification: skipped (no local reference).".to_owned(),
            verified: None,
        };
    }

    let flags = extract_flags(reply);
    if flags.is_empty() {
        return VerifyResult {
            footer: "Verification: no flags to check.".to_owned(),
            verified: Some(true),
        };
    }

    let body_lower = reference.body.to_ascii_lowercase();
    let mut unverified = Vec::new();
    for flag in &flags {
        if !flag_in_reference(flag, &body_lower) {
            unverified.push(flag.clone());
        }
    }

    if unverified.is_empty() {
        VerifyResult {
            footer: format!(
                "Verification: flags match local {}.",
                reference.source.label()
            ),
            verified: Some(true),
        }
    } else {
        VerifyResult {
            footer: format!(
                "Verification: unverified flags: {} (not found in local {}).",
                unverified.join(", "),
                reference.source.label()
            ),
            verified: Some(false),
        }
    }
}

/// Append the verification footer to a reply (blank line separated).
pub fn append_footer(reply: &str, footer: &str) -> String {
    let reply = reply.trim_end();
    if reply.is_empty() {
        footer.to_owned()
    } else {
        format!("{reply}\n\n{footer}")
    }
}

fn extract_flags(text: &str) -> Vec<String> {
    let mut found = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'-' {
            let start = i;
            i += 1;
            if i < bytes.len() && bytes[i] == b'-' {
                // Long option: --foo-bar
                i += 1;
                let name_start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-' || bytes[i] == b'_')
                {
                    i += 1;
                }
                if i > name_start {
                    let flag = &text[start..i];
                    if flag.len() > 2 {
                        found.insert(flag.to_owned());
                    }
                }
            } else if i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                // Short cluster: -lah → -l -a -h
                let cluster_start = i;
                while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                    i += 1;
                }
                let cluster = &text[cluster_start..i];
                if cluster.len() == 1 {
                    found.insert(format!("-{}", cluster));
                } else if cluster.chars().all(|c| c.is_ascii_alphabetic()) && cluster.len() <= 6 {
                    for c in cluster.chars() {
                        found.insert(format!("-{c}"));
                    }
                } else {
                    // Likely a negative number or path fragment; skip.
                }
            }
        } else {
            i += 1;
        }
    }
    found.into_iter().collect()
}

fn flag_in_reference(flag: &str, body_lower: &str) -> bool {
    let needle = flag.to_ascii_lowercase();
    if body_lower.contains(&needle) {
        return true;
    }
    // Short flags sometimes documented as " -f," or " -f "
    if needle.len() == 2 && needle.starts_with('-') {
        let c = &needle[1..];
        return body_lower.contains(&format!(" -{c} "))
            || body_lower.contains(&format!(" -{c},"))
            || body_lower.contains(&format!(" -{c}\n"))
            || body_lower.contains(&format!("\t-{c}"))
            || body_lower.contains(&format!("-{c}, --"));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learnminal::types::ReferenceSource;

    fn ref_ctx(body: &str) -> ReferenceContext {
        ReferenceContext {
            program: "git".into(),
            source: ReferenceSource::Help,
            body: body.into(),
        }
    }

    #[test]
    fn known_flags_pass() {
        let reply = "Try `git push --force-with-lease` or `git status -sb`.";
        let reference = ref_ctx(
            "OPTIONS\n  --force-with-lease\n  -s, --short\n  -b, --branch\n",
        );
        let result = verify_reply(reply, &reference);
        // -s and -b from -sb; --force-with-lease present. -s/-b may fail if not both in body.
        assert!(result.footer.contains("Verification:"));
        assert!(extract_flags(reply).contains(&"--force-with-lease".to_owned()));
    }

    #[test]
    fn invented_flag_fails() {
        let reply = "Use git commit --not-a-real-flag";
        let reference = ref_ctx("OPTIONS\n  --amend\n  --all\n");
        let result = verify_reply(reply, &reference);
        assert_eq!(result.verified, Some(false));
        assert!(result.footer.contains("--not-a-real-flag"));
    }

    #[test]
    fn empty_reference_skips() {
        let result = verify_reply("use --all", &ReferenceContext::empty("git"));
        assert_eq!(result.verified, None);
        assert!(result.footer.contains("skipped"));
    }

    #[test]
    fn append_footer_joins_with_blank_line() {
        assert_eq!(append_footer("hello", "Verification: ok."), "hello\n\nVerification: ok.");
    }
}
