//! ESP32-S3 firmware: loads the model from flash, allocates the state in internal
//! SRAM (spilling into PSRAM for larger models) and provides an interactive
//! console over USB serial. You type the
//! prompt in the monitor (`make flash` or `espflash monitor`) and the board generates
//! the text; commands like `/temp 0.5` change sampling without reflashing.
//!
//! Initialization based on the `esp-generate` 1.4.0 template (esp-hal 1.2).

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec::Vec;
use esp_backtrace as _;
use esp_hal::Blocking;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::rng::Rng;
use esp_hal::time::{Duration, Instant};
use esp_hal::usb::usb_serial_jtag::{UsbSerialJtag, UsbSerialJtagRx};
use esp_println::{print, println};
use tinyllm_core::{Model, Sampler, State, Tokenizer, tokenizer};

// Header required by the esp-idf bootloader; without it espflash refuses to flash.
esp_bootloader_esp_idf::esp_app_desc!();

// Weights and tokenizer embedded in the firmware image (they go to flash and are
// read via XIP, without going through RAM). `include_bytes!` only guarantees
// 1-byte alignment, which is why the parser reads everything with from_le_bytes.
static MODEL_BYTES: &[u8] = include_bytes!(env!("TINYLLM_MODEL"));
static TOKENIZER_BYTES: &[u8] = include_bytes!(env!("TINYLLM_TOKENIZER"));

/// Prompt used when you press Enter without having typed one before.
const DEFAULT_PROMPT: &str = "Once upon a time";
/// KV cache capacity in tokens (prompt + generated text). This is what sets
/// the heap used by the state: ~1.3 KB per token on stories260K.
const MAX_STEPS: usize = 128;
const DEFAULT_TEMPERATURE: f32 = 0.8;
/// Maximum length of a line typed into the console.
const MAX_LINE: usize = 256;

/// Sampling parameters adjustable through console commands.
struct Settings {
    temperature: f32,
    steps: usize,
    /// `None` = new seed for every generation.
    seed: Option<u32>,
    /// Shows how the prompt is split into tokens and separates generated tokens with `|`.
    verbose: bool,
}

#[main]
fn main() -> ! {
    esp_println::logger::init_logger_from_env();

    // Without `CpuClock::max()` the chip may run below 240 MHz.
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // Main heap in internal SRAM. With stories260K and MAX_STEPS=128 the state
    // takes ~168 KB, the KV cache being two contiguous 80 KB blocks.
    // esp-alloc uses the regions in the order they are registered.
    esp_alloc::heap_allocator!(size: 224 * 1024);
    // Bootloader RAM, free after boot: a reserve for small allocations.
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);
    // 8 MB external PSRAM, registered last so it only receives what doesn't fit
    // in SRAM: stories260K stays entirely in SRAM, stories15M spills its KV cache,
    // logits and tokenizer index here. The N16R8 has octal PSRAM (confirmed with
    // `esptool flash-id`); the `Auto` mode probes for it but esp-hal itself calls
    // that detection "not entirely reliable".
    esp_alloc::psram_allocator!(
        peripherals.PSRAM,
        esp_hal::psram,
        esp_hal::psram::PsramConfig {
            mode: esp_hal::psram::PsramMode::OctalSpi,
            ..Default::default()
        }
    );

    // Receive only: output still goes through esp-println, on the same peripheral.
    let (rx, _tx) = UsbSerialJtag::new(peripherals.USB_DEVICE).split();
    let mut console = Console::new(rx);
    let rng = Rng::new();

    // USB-Serial-JTAG re-enumerates on every reset. Wait for the PC's serial monitor
    // to reconnect before printing, otherwise the start of the output is lost.
    let boot = Instant::now();
    while boot.elapsed() < Duration::from_millis(3000) {}

    println!();
    println!("tinyllm-esp32s3");
    println!("model: {} KB in flash", MODEL_BYTES.len() / 1024);

    let model = match Model::from_bytes(MODEL_BYTES) {
        Ok(m) => m,
        Err(e) => {
            println!("model error: {:?}", e);
            loop {}
        }
    };
    let c = model.config;
    println!(
        "dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq={}",
        c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len
    );

    let tok = Tokenizer::from_bytes(TOKENIZER_BYTES, c.vocab_size).expect("tokenizer");
    let mut state = State::new(&c, MAX_STEPS);
    println!("state: {} KB on the heap", state.heap_bytes() / 1024);

    let mut settings = Settings {
        temperature: DEFAULT_TEMPERATURE,
        steps: state.max_seq,
        seed: None,
        verbose: false,
    };
    let mut last_prompt: Vec<u8> = Vec::from(DEFAULT_PROMPT.as_bytes());
    let mut line: Vec<u8> = Vec::with_capacity(MAX_LINE);
    let mut tokens: Vec<usize> = Vec::new();
    let mut piece: Vec<u8> = Vec::new();

    println!();
    print_help(&settings, state.max_seq);

    loop {
        print!("\n> ");
        console.read_line(&mut line);
        // read_line only accepts printable ASCII, so this never fails.
        let text = core::str::from_utf8(&line).unwrap_or("").trim();

        if text.starts_with('/') {
            command(text, &mut settings, state.max_seq);
            continue;
        }
        if !text.is_empty() {
            last_prompt.clear();
            last_prompt.extend_from_slice(text.as_bytes());
        }
        let seed = settings.seed.unwrap_or_else(|| random_seed(&rng));
        generate(
            &model,
            &tok,
            &mut state,
            &last_prompt,
            &settings,
            seed,
            &mut tokens,
            &mut piece,
        );
    }
}

/// Generates text from `prompt` and prints it as the tokens come out.
#[allow(clippy::too_many_arguments)]
fn generate(
    model: &Model,
    tok: &Tokenizer,
    state: &mut State,
    prompt: &[u8],
    s: &Settings,
    seed: u32,
    tokens: &mut Vec<usize>,
    piece: &mut Vec<u8>,
) {
    tok.encode(prompt, true, false, tokens);
    let steps = s.steps.min(state.max_seq);
    println!("[seed {}, temp {:.2}, steps {}]", seed, s.temperature, steps);

    if s.verbose {
        print!("prompt = {} tokens:", tokens.len());
        for &t in tokens.iter() {
            print!(" {}:", t);
            print_piece(tok.vocab[t]);
        }
        println!();
    }
    if tokens.len() >= steps {
        println!(
            "warning: the prompt has {} tokens and /steps is {}; no room left to generate",
            tokens.len(),
            steps
        );
    }

    let mut sampler = Sampler::new(s.temperature, seed as u64);
    let mut token = tokens[0];
    let mut generated = 0usize;
    let t0 = Instant::now();

    for pos in 0..steps {
        let logits = model.forward(state, token, pos);
        let next = if pos + 1 < tokens.len() {
            tokens[pos + 1] // still consuming the prompt
        } else {
            let mut l = logits.to_vec();
            generated += 1;
            sampler.sample(&mut l)
        };
        if next == tokenizer::EOS || next == tokenizer::BOS {
            break;
        }
        piece.clear();
        tok.decode(token, next, piece);
        if s.verbose && pos + 1 >= tokens.len() {
            print!("|");
        }
        if let Ok(p) = core::str::from_utf8(piece) {
            print!("{}", p);
        }
        token = next;
    }

    let ms = t0.elapsed().as_millis();
    // tok/s with two decimal places, using integers
    let rate = generated as u64 * 100_000 / ms.max(1);
    println!();
    println!(
        "-- {} tokens in {} ms ({}.{:02} tok/s)",
        generated,
        ms,
        rate / 100,
        rate % 100
    );
}

/// Handles a line that starts with `/`.
fn command(line: &str, s: &mut Settings, max_steps: usize) {
    let mut parts = line.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let arg = parts.next();
    match (cmd, arg) {
        ("/temp", Some(a)) => match a.parse::<f32>() {
            Ok(t) if (0.0..=2.0).contains(&t) => {
                s.temperature = t;
                println!("temperature = {:.2}", t);
            }
            _ => println!("usage: /temp <0.0 to 2.0>"),
        },
        ("/steps", Some(a)) => match a.parse::<usize>() {
            Ok(n) if (1..=max_steps).contains(&n) => {
                s.steps = n;
                println!("steps = {}", n);
            }
            _ => println!("usage: /steps <1 to {}>", max_steps),
        },
        ("/seed", None) => {
            s.seed = None;
            println!("new seed for every story");
        }
        ("/seed", Some(a)) => match a.parse::<u32>() {
            Ok(v) => {
                s.seed = Some(v);
                println!("fixed seed = {}", v);
            }
            Err(_) => println!("usage: /seed <0 to 4294967295>, or /seed alone for a new one every time"),
        },
        ("/verbose", None) => {
            s.verbose = !s.verbose;
            println!("verbose {}", if s.verbose { "on" } else { "off" });
        }
        ("/heap", None) => println!("{}", esp_alloc::HEAP.stats()),
        ("/help", None) | ("/?", None) => print_help(s, max_steps),
        _ => println!("unknown command: {} (type /help)", line),
    }
}

fn print_help(s: &Settings, max_steps: usize) {
    println!("Type the start of a story in English and press Enter.");
    println!("Enter on an empty line repeats the last prompt.");
    println!();
    println!("  /temp <t>    0 = always the most likely word; 1.2+ = chaotic  [{:.2}]", s.temperature);
    println!("  /steps <n>   total tokens, prompt included (1 to {})         [{}]", max_steps, s.steps);
    match s.seed {
        Some(v) => println!("  /seed [n]    fixed seed; /seed alone = new one every story    [{}]", v),
        None => println!("  /seed [n]    fixed seed; /seed alone = new one every story    [random]"),
    }
    println!(
        "  /verbose     show the tokens the model sees                   [{}]",
        if s.verbose { "on" } else { "off" }
    );
    println!("  /heap        memory usage");
    println!("  /help        this help");
    println!("Ctrl+R resets the board, Ctrl+C exits the monitor.");
}

/// Prints a vocabulary token in readable form: `"Once"`, `"\n"`, `<0x0A>`.
fn print_piece(p: &[u8]) {
    print!("\"");
    for &b in p {
        match b {
            b'\n' => print!("\\n"),
            b'"' => print!("\\\""),
            0x20..=0x7e => print!("{}", b as char),
            _ => print!("\\x{:02x}", b),
        }
    }
    print!("\"");
}

/// 32-bit seed (easy to retype with `/seed`).
fn random_seed(rng: &Rng) -> u32 {
    // With Wi-Fi/BT off the hardware RNG is only pseudorandom; the moment
    // you pressed Enter, in microseconds, supplies the rest of the entropy.
    let t = Instant::now().duration_since_epoch().as_micros();
    rng.random() ^ (t as u32) ^ ((t >> 32) as u32)
}

/// USB-Serial-JTAG line reader with echo and backspace.
struct Console<'d> {
    rx: UsbSerialJtagRx<'d, Blocking>,
    /// The last byte was `\r` (to treat `\r\n` as a single Enter).
    last_cr: bool,
    /// Escape sequence state (arrow keys etc.): 0 outside, 1 after ESC, 2 inside.
    esc: u8,
}

impl<'d> Console<'d> {
    fn new(rx: UsbSerialJtagRx<'d, Blocking>) -> Self {
        Console { rx, last_cr: false, esc: 0 }
    }

    /// Blocks until Enter. Accepts only printable ASCII; ignores arrow keys.
    fn read_line(&mut self, line: &mut Vec<u8>) {
        line.clear();
        loop {
            let Ok(b) = self.rx.read_byte() else { continue };

            // Discards escape sequences: ESC [ ... <letter> or ESC O <letter>
            match self.esc {
                1 => {
                    self.esc = if b == b'[' || b == b'O' { 2 } else { 0 };
                    continue;
                }
                2 => {
                    if (0x40..=0x7e).contains(&b) {
                        self.esc = 0;
                    }
                    continue;
                }
                _ => {}
            }

            let after_cr = core::mem::replace(&mut self.last_cr, b == b'\r');
            match b {
                b'\n' if after_cr => {}
                b'\r' | b'\n' => {
                    println!();
                    return;
                }
                0x08 | 0x7f => {
                    if line.pop().is_some() {
                        print!("\x08 \x08");
                    }
                }
                0x1b => self.esc = 1,
                0x20..=0x7e if line.len() < MAX_LINE => {
                    line.push(b);
                    print!("{}", b as char);
                }
                _ => {}
            }
        }
    }
}
