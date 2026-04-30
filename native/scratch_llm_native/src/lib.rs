use fancy_regex::Regex;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const DEFAULT_SPECIAL_TOKEN: &str = "<|endoftext|>";
const PRETOKENIZER_PATTERN: &str =
    r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

#[derive(Debug, Serialize, Deserialize)]
struct TokenizerJson {
    special_tokens: Vec<String>,
    vocab: Vec<(usize, String)>,
    merges: Vec<(String, String)>,
}

#[derive(Debug)]
struct Tokenizer {
    vocab: HashMap<usize, Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, usize>,
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
    special_tokens: Vec<String>,
    special_token_to_id: HashMap<String, usize>,
    special_token_ids: HashMap<usize, ()>,
}

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn py_io_error(path: &str, err: std::io::Error) -> PyErr {
    PyIOError::new_err(format!("{path}: {err}"))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_to_bytes(hex: &str) -> PyResult<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(py_value_error("hex string has odd length"));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for idx in (0..bytes.len()).step_by(2) {
        let hi = (bytes[idx] as char)
            .to_digit(16)
            .ok_or_else(|| py_value_error("invalid hex digit"))?;
        let lo = (bytes[idx + 1] as char)
            .to_digit(16)
            .ok_or_else(|| py_value_error("invalid hex digit"))?;
        out.push(((hi << 4) | lo) as u8);
    }
    Ok(out)
}

fn split_on_special_tokens(text: &str, special_tokens: &[String]) -> Vec<(String, bool)> {
    if special_tokens.is_empty() {
        return vec![(text.to_string(), false)];
    }

    let mut pieces = Vec::new();
    let mut offset = 0;
    while offset < text.len() {
        let remaining = &text[offset..];
        let mut best_match: Option<&String> = None;
        for special in special_tokens {
            if remaining.starts_with(special)
                && best_match.map_or(true, |current| special.len() > current.len())
            {
                best_match = Some(special);
            }
        }
        if let Some(special) = best_match {
            pieces.push((special.clone(), true));
            offset += special.len();
            continue;
        }

        let next_special = special_tokens
            .iter()
            .filter_map(|special| remaining.find(special))
            .min()
            .unwrap_or(remaining.len());
        if next_special > 0 {
            pieces.push((remaining[..next_special].to_string(), false));
            offset += next_special;
        } else {
            // This should be unreachable because starts_with is handled above.
            offset += remaining.chars().next().map_or(1, |ch| ch.len_utf8());
        }
    }
    pieces
}

fn pretokenize(text: &str, special_tokens: &[String]) -> PyResult<HashMap<Vec<Vec<u8>>, usize>> {
    let pattern = Regex::new(PRETOKENIZER_PATTERN).map_err(|err| py_value_error(err.to_string()))?;
    let mut counts: HashMap<Vec<Vec<u8>>, usize> = HashMap::new();
    for (piece, is_special) in split_on_special_tokens(text, special_tokens) {
        if is_special {
            continue;
        }
        for maybe_match in pattern.find_iter(&piece) {
            let matched = maybe_match.map_err(|err| py_value_error(err.to_string()))?;
            let word: Vec<Vec<u8>> = matched
                .as_str()
                .as_bytes()
                .iter()
                .map(|byte| vec![*byte])
                .collect();
            *counts.entry(word).or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn pair_counts(word_counts: &HashMap<Vec<Vec<u8>>, usize>) -> HashMap<(Vec<u8>, Vec<u8>), usize> {
    let mut counts = HashMap::new();
    for (word, count) in word_counts {
        for pair in word.windows(2) {
            let key = (pair[0].clone(), pair[1].clone());
            *counts.entry(key).or_insert(0) += *count;
        }
    }
    counts
}

fn merge_word(word: &[Vec<u8>], pair: &(Vec<u8>, Vec<u8>)) -> Vec<Vec<u8>> {
    let mut merged = Vec::with_capacity(word.len());
    let mut idx = 0;
    while idx < word.len() {
        if idx + 1 < word.len() && word[idx] == pair.0 && word[idx + 1] == pair.1 {
            let mut token = pair.0.clone();
            token.extend_from_slice(&pair.1);
            merged.push(token);
            idx += 2;
        } else {
            merged.push(word[idx].clone());
            idx += 1;
        }
    }
    merged
}

fn train_bpe_impl(
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<(HashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>)> {
    if vocab_size < 256 + special_tokens.len() {
        return Err(py_value_error(
            "vocab_size must fit all byte tokens plus special tokens",
        ));
    }

    let text = fs::read_to_string(input_path).map_err(|err| py_io_error(input_path, err))?;
    let mut word_counts = pretokenize(&text, &special_tokens)?;
    let mut vocab: HashMap<usize, Vec<u8>> = (0..=255).map(|idx| (idx, vec![idx as u8])).collect();

    for special in &special_tokens {
        let bytes = special.as_bytes().to_vec();
        if !vocab.values().any(|token| *token == bytes) {
            vocab.insert(vocab.len(), bytes);
        }
    }

    let mut merges = Vec::new();
    while vocab.len() < vocab_size {
        let pairs = pair_counts(&word_counts);
        if pairs.is_empty() {
            break;
        }
        let best_pair = pairs
            .iter()
            .max_by(|(left_pair, left_count), (right_pair, right_count)| {
                left_count
                    .cmp(right_count)
                    .then_with(|| left_pair.cmp(right_pair))
            })
            .map(|(pair, _count)| pair.clone())
            .expect("pairs is non-empty");

        let mut merged_token = best_pair.0.clone();
        merged_token.extend_from_slice(&best_pair.1);
        vocab.insert(vocab.len(), merged_token);
        merges.push(best_pair.clone());

        let mut next_counts: HashMap<Vec<Vec<u8>>, usize> = HashMap::new();
        for (word, count) in &word_counts {
            *next_counts.entry(merge_word(word, &best_pair)).or_insert(0) += *count;
        }
        word_counts = next_counts;
    }

    Ok((vocab, merges))
}

fn save_tokenizer(
    path: &str,
    vocab: HashMap<usize, Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    special_tokens: Vec<String>,
) -> PyResult<()> {
    let mut vocab_rows: Vec<(usize, String)> = vocab
        .into_iter()
        .map(|(idx, token)| (idx, bytes_to_hex(&token)))
        .collect();
    vocab_rows.sort_by_key(|(idx, _)| *idx);
    let merge_rows = merges
        .into_iter()
        .map(|(left, right)| (bytes_to_hex(&left), bytes_to_hex(&right)))
        .collect();
    let payload = TokenizerJson {
        special_tokens,
        vocab: vocab_rows,
        merges: merge_rows,
    };
    let json = serde_json::to_string(&payload).map_err(|err| py_value_error(err.to_string()))?;
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|err| py_io_error(path, err))?;
    }
    fs::write(path, json).map_err(|err| py_io_error(path, err))?;
    Ok(())
}

fn load_tokenizer(path: &str) -> PyResult<Tokenizer> {
    let text = fs::read_to_string(path).map_err(|err| py_io_error(path, err))?;
    let payload: TokenizerJson =
        serde_json::from_str(&text).map_err(|err| py_value_error(err.to_string()))?;

    let mut vocab = HashMap::new();
    for (idx, token_hex) in payload.vocab {
        vocab.insert(idx, hex_to_bytes(&token_hex)?);
    }
    let token_to_id = vocab
        .iter()
        .map(|(idx, token)| (token.clone(), *idx))
        .collect::<HashMap<_, _>>();

    let mut merge_ranks = HashMap::new();
    for (rank, (left_hex, right_hex)) in payload.merges.into_iter().enumerate() {
        merge_ranks.insert((hex_to_bytes(&left_hex)?, hex_to_bytes(&right_hex)?), rank);
    }
    let special_token_to_id = payload
        .special_tokens
        .iter()
        .filter_map(|special| {
            token_to_id
                .get(special.as_bytes())
                .map(|idx| (special.clone(), *idx))
        })
        .collect::<HashMap<_, _>>();
    let special_token_ids = special_token_to_id.keys().filter_map(|special| {
        token_to_id
            .get(special.as_bytes())
            .map(|idx| (*idx, ()))
    }).collect::<HashMap<_, _>>();

    Ok(Tokenizer {
        vocab,
        token_to_id,
        merge_ranks,
        special_tokens: payload.special_tokens,
        special_token_to_id,
        special_token_ids,
    })
}

fn encode_pretoken(tokenizer: &Tokenizer, token: &[u8]) -> PyResult<Vec<usize>> {
    let mut pieces: Vec<Vec<u8>> = token.iter().map(|byte| vec![*byte]).collect();
    while pieces.len() > 1 {
        let mut best_pair: Option<((Vec<u8>, Vec<u8>), usize)> = None;
        for pair in pieces.windows(2) {
            let candidate = (pair[0].clone(), pair[1].clone());
            if let Some(rank) = tokenizer.merge_ranks.get(&candidate) {
                if best_pair.as_ref().map_or(true, |(_, best_rank)| rank < best_rank) {
                    best_pair = Some((candidate, *rank));
                }
            }
        }
        let Some((pair, _rank)) = best_pair else {
            break;
        };
        pieces = merge_word(&pieces, &pair);
    }
    pieces
        .iter()
        .map(|piece| {
            tokenizer
                .token_to_id
                .get(piece)
                .copied()
                .ok_or_else(|| py_value_error("token piece missing from vocabulary"))
        })
        .collect()
}

#[pyfunction]
fn backend_name() -> &'static str {
    "rust-pyo3"
}

#[pyfunction]
#[pyo3(signature = (input_path, vocab_size, output_path, special_tokens=None))]
fn train_bpe_to_file(
    input_path: String,
    vocab_size: usize,
    output_path: String,
    special_tokens: Option<Vec<String>>,
) -> PyResult<(usize, usize)> {
    let special_tokens = special_tokens.unwrap_or_else(|| vec![DEFAULT_SPECIAL_TOKEN.to_string()]);
    let (vocab, merges) = train_bpe_impl(&input_path, vocab_size, special_tokens.clone())?;
    let vocab_len = vocab.len();
    let merges_len = merges.len();
    save_tokenizer(&output_path, vocab, merges, special_tokens)?;
    Ok((vocab_len, merges_len))
}

#[pyfunction]
fn encode_file(
    tokenizer_path: String,
    text: String,
    add_special_tokens: bool,
) -> PyResult<Vec<usize>> {
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let pattern = Regex::new(PRETOKENIZER_PATTERN).map_err(|err| py_value_error(err.to_string()))?;
    let mut ids = Vec::new();

    for (piece, is_special) in split_on_special_tokens(&text, &tokenizer.special_tokens) {
        if is_special {
            let Some(id) = tokenizer.special_token_to_id.get(&piece) else {
                return Err(py_value_error(format!("unknown special token {piece:?}")));
            };
            ids.push(*id);
            continue;
        }
        for maybe_match in pattern.find_iter(&piece) {
            let matched = maybe_match.map_err(|err| py_value_error(err.to_string()))?;
            ids.extend(encode_pretoken(&tokenizer, matched.as_str().as_bytes())?);
        }
    }

    if add_special_tokens {
        if let Some(id) = tokenizer.special_token_to_id.get(DEFAULT_SPECIAL_TOKEN) {
            ids.push(*id);
        }
    }
    Ok(ids)
}

#[pyfunction]
fn decode_file(
    tokenizer_path: String,
    ids: Vec<usize>,
    skip_special_tokens: bool,
) -> PyResult<String> {
    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let mut bytes = Vec::new();
    for id in ids {
        if skip_special_tokens && tokenizer.special_token_ids.contains_key(&id) {
            continue;
        }
        let Some(token) = tokenizer.vocab.get(&id) else {
            return Err(py_value_error(format!("unknown token id {id}")));
        };
        bytes.extend_from_slice(token);
    }
    String::from_utf8(bytes).map_err(|err| py_value_error(err.to_string()))
}

#[pyfunction]
fn tokenizer_vocab_size(tokenizer_path: String) -> PyResult<usize> {
    Ok(load_tokenizer(&tokenizer_path)?.vocab.len())
}

#[pymodule]
fn scratch_llm_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(encode_file, m)?)?;
    m.add_function(wrap_pyfunction!(decode_file, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer_vocab_size, m)?)?;
    Ok(())
}
