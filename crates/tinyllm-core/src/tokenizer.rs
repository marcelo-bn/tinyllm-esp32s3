//! BPE tokenizer compatible with llama2.c's `tokenizer.bin`.
//!
//! Format: `max_token_length: i32`, then for each vocabulary token
//! `score: f32`, `len: i32`, `bytes[len]`.

use alloc::vec::Vec;

use crate::Error;

pub const BOS: usize = 1;
pub const EOS: usize = 2;

pub struct Tokenizer<'a> {
    pub vocab: Vec<&'a [u8]>,
    pub scores: Vec<f32>,
    pub max_token_len: usize,
}

impl<'a> Tokenizer<'a> {
    pub fn from_bytes(buf: &'a [u8], vocab_size: usize) -> Result<Self, Error> {
        let mut pos = 0usize;
        let rd_u32 = |pos: &mut usize| -> Result<u32, Error> {
            let b = buf.get(*pos..*pos + 4).ok_or(Error::Truncated)?;
            *pos += 4;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let max_token_len = rd_u32(&mut pos)? as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            scores.push(f32::from_bits(rd_u32(&mut pos)?));
            let len = rd_u32(&mut pos)? as usize;
            let s = buf.get(pos..pos + len).ok_or(Error::Truncated)?;
            pos += len;
            vocab.push(s);
        }
        Ok(Tokenizer { vocab, scores, max_token_len })
    }

    /// Bytes of token `t`, already handling `<0xNN>` (literal byte) and the leading
    /// space SentencePiece adds after BOS.
    pub fn decode(&self, prev: usize, t: usize, out: &mut Vec<u8>) {
        let mut piece = self.vocab[t];
        if prev == BOS && piece.first() == Some(&b' ') {
            piece = &piece[1..];
        }
        if let Some(b) = parse_byte_token(piece) {
            out.push(b);
        } else {
            out.extend_from_slice(piece);
        }
    }

    fn lookup(&self, s: &[u8]) -> Option<usize> {
        // linear search: vocabularies here have at most 32k entries and
        // encode only runs on the prompt. Switch to a hash if it becomes a bottleneck.
        self.vocab.iter().position(|v| *v == s)
    }

    /// Encodes `text` into tokens (greedy BPE by score, as in llama2.c).
    pub fn encode(&self, text: &[u8], bos: bool, eos: bool, out: &mut Vec<usize>) {
        out.clear();
        if bos {
            out.push(BOS);
        }
        // " " prefix (SentencePiece dummy prefix)
        if !text.is_empty() {
            if let Some(t) = self.lookup(b" ") {
                out.push(t);
            }
        }
        // each byte (or UTF-8 sequence) becomes an initial token
        let mut i = 0;
        while i < text.len() {
            let len = utf8_len(text[i]).min(text.len() - i);
            let chunk = &text[i..i + len];
            match self.lookup(chunk) {
                Some(t) => out.push(t),
                None => {
                    for &b in chunk {
                        out.push(b as usize + 3); // <0xNN> tokens start at 3
                    }
                }
            }
            i += len;
        }

        // merges: join the adjacent pair with the highest score until none are left
        let mut buf: Vec<u8> = Vec::with_capacity(self.max_token_len * 2 + 2);
        loop {
            let mut best: Option<(f32, usize, usize)> = None; // (score, id, idx)
            for idx in 0..out.len().saturating_sub(1) {
                buf.clear();
                buf.extend_from_slice(self.vocab[out[idx]]);
                buf.extend_from_slice(self.vocab[out[idx + 1]]);
                if let Some(id) = self.lookup(&buf) {
                    let sc = self.scores[id];
                    if best.map_or(true, |(bs, _, _)| sc > bs) {
                        best = Some((sc, id, idx));
                    }
                }
            }
            match best {
                None => break,
                Some((_, id, idx)) => {
                    out[idx] = id;
                    out.remove(idx + 1);
                }
            }
        }
        if eos {
            out.push(EOS);
        }
    }
}

fn parse_byte_token(p: &[u8]) -> Option<u8> {
    // "<0xNN>"
    if p.len() == 6 && p.starts_with(b"<0x") && p[5] == b'>' {
        let hi = hex(p[3])?;
        let lo = hex(p[4])?;
        Some(hi << 4 | lo)
    } else {
        None
    }
}

fn hex(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}
