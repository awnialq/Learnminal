//! DuckDuckGo HTML lite web search for the `web_search` Ollama tool.
//!
//! No API key. Scrapes `https://html.duckduckgo.com/html/?q=...` for the top
//! result cards (title, URL, snippet). Soft 8s timeout; failures become a short
//! tool-error string so the model can still answer.

use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};

const SEARCH_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_HITS: usize = 5;
const RESULT_BUDGET_CHARS: usize = 3_000;
const DDG_HTML_URL: &str = "https://html.duckduckgo.com/html/";
const USER_AGENT_VALUE: &str =
    "Mozilla/5.0 (compatible; Learnminal/1.0; +https://github.com/learnminal)";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchError {
    EmptyQuery,
    Http(String),
    NoResults,
}

impl SearchError {
    pub fn as_tool_message(&self) -> String {
        match self {
            SearchError::EmptyQuery => "web_search error: empty query".to_owned(),
            SearchError::Http(msg) => format!("web_search error: {msg}"),
            SearchError::NoResults => "web_search: no results found".to_owned(),
        }
    }
}

/// Run a DuckDuckGo HTML search and return up to five hits.
pub fn search(query: &str) -> Result<Vec<SearchHit>, SearchError> {
    let query = query.trim();
    if query.is_empty() {
        return Err(SearchError::EmptyQuery);
    }

    let client = Client::builder()
        .timeout(SEARCH_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(5))
        .default_headers(default_headers())
        .build()
        .map_err(|err| SearchError::Http(err.to_string()))?;

    let response = client
        .post(DDG_HTML_URL)
        .form(&[("q", query), ("b", "")])
        .send()
        .map_err(|err| SearchError::Http(err.to_string()))?;

    if !response.status().is_success() {
        return Err(SearchError::Http(format!("HTTP {}", response.status())));
    }

    let html = response.text().map_err(|err| SearchError::Http(err.to_string()))?;
    let hits = parse_ddg_html(&html);
    if hits.is_empty() {
        Err(SearchError::NoResults)
    } else {
        Ok(hits)
    }
}

fn default_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Ok(value) = HeaderValue::from_str(USER_AGENT_VALUE) {
        headers.insert(USER_AGENT, value);
    }
    headers
}

/// Parse DuckDuckGo HTML lite results into structured hits.
pub fn parse_ddg_html(html: &str) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut rest = html;

    while hits.len() < MAX_HITS {
        let Some(anchor_start) = find_ci(rest, "class=\"result__a\"")
            .or_else(|| find_ci(rest, "class='result__a'"))
        else {
            break;
        };

        let before = &rest[..anchor_start];
        let tag_start = before.rfind('<').unwrap_or(0);
        let from_tag = &rest[tag_start..];

        let Some(href) = extract_href(from_tag) else {
            rest = &rest[anchor_start + 10..];
            continue;
        };
        let url = normalize_ddg_url(&href);
        if url.is_empty() || url.starts_with("javascript:") {
            rest = &rest[anchor_start + 10..];
            continue;
        }

        let Some(title) = extract_anchor_text(from_tag) else {
            rest = &rest[anchor_start + 10..];
            continue;
        };
        let title = decode_basic_entities(&strip_tags(&title)).trim().to_owned();
        if title.is_empty() {
            rest = &rest[anchor_start + 10..];
            continue;
        }

        // Snippet usually follows in the same result block.
        let after_anchor = from_tag.find("</a>").map(|i| &from_tag[i..]).unwrap_or(from_tag);
        let snippet = extract_snippet(after_anchor).unwrap_or_default();

        hits.push(SearchHit { title, url, snippet });
        rest = &rest[anchor_start + 10..];
    }

    hits
}

fn extract_snippet(block: &str) -> Option<String> {
    let marker = find_ci(block, "result__snippet")?;
    let tag_start = block[..marker].rfind('<')?;
    let from_tag = &block[tag_start..];
    let text = if from_tag.to_ascii_lowercase().starts_with("<a") {
        extract_anchor_text(from_tag)?
    } else {
        let after_gt = from_tag.find('>')?;
        let inner = &from_tag[after_gt + 1..];
        let end = inner.find('<').unwrap_or(inner.len());
        inner[..end].to_owned()
    };
    let text = decode_basic_entities(&strip_tags(&text)).trim().to_owned();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn extract_href(tag_and_rest: &str) -> Option<String> {
    let lower = tag_and_rest.to_ascii_lowercase();
    let href_idx = lower.find("href=")?;
    let after = &tag_and_rest[href_idx + 5..];
    let quote = after.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = &after[1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_owned())
}

fn extract_anchor_text(tag_and_rest: &str) -> Option<String> {
    let gt = tag_and_rest.find('>')?;
    let after = &tag_and_rest[gt + 1..];
    let end = after.find("</a>").or_else(|| after.find("</A>"))?;
    Some(after[..end].to_owned())
}

/// DuckDuckGo wraps redirects as `//duckduckgo.com/l/?uddg=<urlencoded>`.
fn normalize_ddg_url(raw: &str) -> String {
    let raw = raw.trim();
    if let Some(idx) = raw.find("uddg=") {
        let encoded = &raw[idx + 5..];
        let encoded = encoded.split('&').next().unwrap_or(encoded);
        return urlencoding_decode(encoded);
    }
    if raw.starts_with("//") {
        return format!("https:{raw}");
    }
    raw.to_owned()
}

fn urlencoding_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            },
            b'%' if i + 2 < bytes.len() => {
                let hex = &input[i + 1..i + 3];
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            },
            b => {
                out.push(b);
                i += 1;
            },
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {},
        }
    }
    out
}

fn decode_basic_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_ascii_lowercase().find(&needle.to_ascii_lowercase()).map(|idx| {
        // Map byte index from lowercased string — ASCII-only needles/markers.
        idx
    })
}

/// Compact plain-text tool result for the model (under ~3k chars).
pub fn format_tool_result(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return SearchError::NoResults.as_tool_message();
    }
    let mut out = String::from("Web search results:\n");
    for (i, hit) in hits.iter().take(MAX_HITS).enumerate() {
        let entry = format!(
            "{}. {}\n   URL: {}\n   {}\n",
            i + 1,
            hit.title.trim(),
            hit.url.trim(),
            hit.snippet.trim()
        );
        if out.chars().count() + entry.chars().count() > RESULT_BUDGET_CHARS {
            break;
        }
        out.push_str(&entry);
    }
    out
}

/// Search and format in one step for tool handlers.
pub fn search_tool_result(query: &str) -> String {
    match search(query) {
        Ok(hits) => format_tool_result(&hits),
        Err(err) => err.as_tool_message(),
    }
}

/// Whether web search tooling is enabled (`LEARNMINAL_WEB_SEARCH` not `0`/`false`/`off`).
pub fn web_search_enabled() -> bool {
    match std::env::var("LEARNMINAL_WEB_SEARCH") {
        Ok(value) => {
            let v = value.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        },
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"
<html><body>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://example.com/git-rebase">Git rebase docs</a>
  <a class="result__snippet" href="https://example.com/git-rebase">How to use git rebase interactively.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fkernel.org%2Fman%2Fgit&amp;rut=abc">man git</a>
  <td class="result__snippet">The git man page on kernel.org</td>
</div>
<div class="result results_links">
  <a class="result__a" href="javascript:void(0)">Ignore me</a>
</div>
</body></html>
"#;

    #[test]
    fn parse_fixture_extracts_titles_urls_snippets() {
        let hits = parse_ddg_html(FIXTURE);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "Git rebase docs");
        assert_eq!(hits[0].url, "https://example.com/git-rebase");
        assert!(hits[0].snippet.contains("git rebase"));
        assert_eq!(hits[1].title, "man git");
        assert_eq!(hits[1].url, "https://kernel.org/man/git");
        assert!(hits[1].snippet.contains("man page"));
    }

    #[test]
    fn format_tool_result_lists_hits() {
        let hits = parse_ddg_html(FIXTURE);
        let text = format_tool_result(&hits);
        assert!(text.contains("Web search results:"));
        assert!(text.contains("Git rebase docs"));
        assert!(text.contains("https://example.com/git-rebase"));
    }

    #[test]
    fn format_empty_is_error_message() {
        assert_eq!(format_tool_result(&[]), SearchError::NoResults.as_tool_message());
    }

    #[test]
    fn empty_query_errors() {
        assert_eq!(search("   "), Err(SearchError::EmptyQuery));
    }

    #[test]
    fn urlencoding_decode_basic() {
        assert_eq!(urlencoding_decode("https%3A%2F%2Fex.com%2Fa+b"), "https://ex.com/a b");
    }
}
