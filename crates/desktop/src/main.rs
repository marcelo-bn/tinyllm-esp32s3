//! Desktop CLI. Same inference crate as the firmware; only main differs.
//!
//! Usage:
//!   tinyllm <model.tlm> <tokenizer.bin> [--prompt "text"] [--steps N] [--temp T] [--seed S]
//!
//! With --temp 0 the output is deterministic: compare it with `tools/reference.py`.

use std::io::Write;
use std::time::Instant;

use tinyllm_core::{tokenizer, Model, Sampler, State, Tokenizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tinyllm <model.tlm> <tokenizer.bin> [--prompt T] [--steps N] [--temp T] [--seed S]");
        std::process::exit(1);
    }

    let mut prompt = String::new();
    let mut steps = 128usize;
    let mut temp = 0.0f32;
    let mut seed = 42u64;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => { prompt = args[i + 1].clone(); i += 2; }
            "--steps" => { steps = args[i + 1].parse().unwrap(); i += 2; }
            "--temp" => { temp = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown argument: {other}"); std::process::exit(1); }
        }
    }

    let model_bytes = std::fs::read(&args[1]).expect("reading model");
    let tok_bytes = std::fs::read(&args[2]).expect("reading tokenizer");

    let model = Model::from_bytes(&model_bytes).expect("parsing model");
    let c = model.config;
    eprintln!(
        "model: dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq={}",
        c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len
    );

    let tok = Tokenizer::from_bytes(&tok_bytes, c.vocab_size).expect("parsing tokenizer");
    let steps = steps.min(c.seq_len);
    let mut state = State::new(&c, steps);
    eprintln!("state: {} KB on the heap", state.heap_bytes() / 1024);

    let mut tokens = Vec::new();
    tok.encode(prompt.as_bytes(), true, false, &mut tokens);

    let mut sampler = Sampler::new(temp, seed);
    let mut out = std::io::stdout();
    let mut buf = Vec::new();
    let mut token = tokens[0];
    let t0 = Instant::now();
    let mut generated = 0usize;

    for pos in 0..steps {
        let logits = model.forward(&mut state, token, pos);
        let next = if pos + 1 < tokens.len() {
            tokens[pos + 1] // still consuming the prompt
        } else {
            // we need &mut: copy the logits (small vocab, cheap)
            let mut l = logits.to_vec();
            generated += 1;
            sampler.sample(&mut l)
        };
        if next == tokenizer::EOS || next == tokenizer::BOS {
            break;
        }
        buf.clear();
        tok.decode(token, next, &mut buf);
        out.write_all(&buf).unwrap();
        out.flush().unwrap();
        token = next;
    }
    println!();
    let dt = t0.elapsed().as_secs_f32();
    eprintln!("{generated} tokens generated in {dt:.2}s ({:.1} tok/s)", generated as f32 / dt);
}
