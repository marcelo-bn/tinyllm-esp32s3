# tinyllm-s3

This guide takes you from a fresh machine to a tiny language model generating
text on an ESP32-S3 N16R8 (16 MB flash, 8 MB PSRAM). Follow the parts in order:
each one ends with a check that has to pass before you move on.

| Part | What you do | Board needed? |
|---|---|---|
| **1. Set up your computer** | Install the toolchains, convert the model, validate it on your PC | No |
| **2. Build the firmware** | Compile the firmware and check that the image fits the chip | No |
| **3. Flash and run on the board** | Connect the board, flash it, and watch it generate text | Yes |
| **4. Play with the board** | Type prompts, change temperature and seed, see the tokens, add debug output | Yes |
| **5. Run a bigger model** | Switch to stories15M, which uses the PSRAM | Yes |

Tested on **Ubuntu 24.04** in October 2026. Other Linux distributions should
work with their own package names. macOS and Windows are not covered.

---

## Part 1 — Set up your computer

You will install two separate worlds that meet in this repository:

```
 Python side (uv)                       Rust side (rustup + espup)
 ────────────────                       ──────────────────────────
 tools/export.py   converts the model   crates/        runs the model on your PC
 tools/reference.py checks the Rust     firmware/      runs the model on the chip
```

At the end of this part you will have generated text on your PC with the same
Rust code that later runs on the board.

### What gets installed

| Tool | Why you need it |
|---|---|
| `build-essential`, `pkg-config`, `libudev-dev` | C linker for Rust, and the USB/serial library `espflash` links against |
| Rust **stable ≥ 1.95** (via `rustup`) | Builds the desktop CLI and the ESP tools below |
| `espup` | Installs the `esp` toolchain: a Rust fork that can target the ESP32-S3's Xtensa CPU, which upstream Rust does not support |
| `espflash` | Flashes firmware to the board and opens the serial monitor |
| `esp-generate` | Generates a reference firmware project with up-to-date crate versions (used in Part 2) |
| `uv` | Manages Python and the `numpy` dependency for the scripts in `tools/` |

Plan for roughly **2 GB of disk** (the `esp` toolchain alone is about 1.7 GB)
and **10–15 minutes**, mostly spent compiling the Cargo tools.

### Step 1 — System packages

```bash
sudo apt update
sudo apt install build-essential pkg-config libudev-dev curl git
```

### Step 2 — Serial port access

To talk to the board without `sudo`, your user must be in the `dialout` group:

```bash
groups | grep -q dialout && echo "already in dialout" || sudo usermod -aG dialout "$USER"
```

If the command added you to the group, **log out and back in** (or reboot) for it
to take effect.

### Step 3 — Rust

If you don't have Rust yet:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
```

If you already have it, update it. The current `esp-generate` refuses to build
on anything older than 1.95:

```bash
rustup update stable
rustc --version          # must print 1.95.0 or newer
```

### Step 4 — ESP tools

```bash
cargo install espup espflash esp-generate --locked
```

This compiles all three from source and takes a few minutes. `--locked` uses the
exact dependency versions each tool was released with, which avoids surprise
build failures.

### Step 5 — The Xtensa toolchain

```bash
espup install --targets esp32s3
```

This downloads a Rust compiler, LLVM and GCC for Xtensa into
`~/.rustup/toolchains/esp`, and writes `~/export-esp.sh`. That script sets two
environment variables the firmware build needs (`PATH` for the
`xtensa-esp32s3-elf-gcc` linker, and `LIBCLANG_PATH`).

**You must load it in every new terminal** before building the firmware:

```bash
. ~/export-esp.sh
```

To make that automatic, add the same line to your `~/.bashrc`.

You never have to select the toolchain by hand.
`firmware/esp32s3/rust-toolchain.toml` pins it to `esp`, so `cargo` switches to
it automatically inside that folder and uses the normal stable toolchain
everywhere else.

### Step 6 — Python with uv

If you don't have `uv` yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repository root:

```bash
make setup               # same as: uv sync
```

`uv` reads `.python-version` (Python 3.13), downloads that interpreter if you
don't have it, and creates `.venv/` with the exact `numpy` version pinned in
`uv.lock`. You never need to activate the virtual environment: the Makefile runs
every script through `uv run`.

### Step 7 — Download and convert the model

Start with **stories260K**, a 260-thousand-parameter model trained on short
children's stories. It runs entirely from the chip's fast internal RAM and
generates 20 to 35 tokens per second on the board. Part 5 moves up to the
60× larger stories15M once everything works.

```bash
make models
```

This downloads both models with their tokenizers into `models/` (about 77 MB,
almost all of it stories15M) and runs `tools/export.py` on each. The script
converts the float32 weights into the project's `.tlm` format: int8 weights
plus one float scale per row. Among the download progress bars you should see:

```
config: {'dim': 64, 'hidden_dim': 172, 'n_layers': 5, 'n_heads': 8, 'n_kv_heads': 4, 'vocab_size': 512, 'seq_len': 512}  shared_cls=True
0.26 M quantized parameters -> 270 KB in models/stories260K.tlm
KV cache at seq_len=512: 640 KB  (at seq=256: 320 KB)
...
config: {'dim': 288, 'hidden_dim': 768, 'n_layers': 6, 'n_heads': 6, 'n_kv_heads': 6, 'vocab_size': 32000, 'seq_len': 256}  shared_cls=True
15.19 M quantized parameters -> 15041 KB in models/stories15M.tlm
KV cache at seq_len=256: 3456 KB  (at seq=256: 3456 KB)
```

`make models` only fetches or converts what is missing or out of date, so it
is safe to run again: a second run prints `Nothing to be done`. If a download
is interrupted, running it again retries that file.

### Step 8 — Validate on your PC

This is the most important check in the whole guide. The PC build uses
**exactly the same inference code** (`crates/tinyllm-core`) as the firmware, so
if it works here, any later problem is in the board setup, not in the model.

```bash
make desktop
```

Expected output (the story text is deterministic because the Makefile uses
temperature 0):

```
model: dim=64 hidden=172 layers=5 heads=8 kv_heads=4 vocab=512 seq=512
state: 86 KB on the heap
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high. She as
60 tokens generated in 0.01s (6627.1 tok/s)
```

Then compare the Rust output against the independent numpy implementation:

```bash
make check
```

It must print:

```
OK: Rust == Python
```

For stories260K the two outputs must match exactly. If they differ, the bug is
in `crates/tinyllm-core`. The usual suspects are RoPE, grouped-query
attention, or the order of quantization steps.

With larger models, a difference that appears **late** in the text is not
necessarily a bug. The Rust and numpy code round in slightly different ways,
and when the two most likely next tokens score almost the same, that rounding
can tip the choice (Part 5 shows an example).

### Part 1 checklist

Open a **new terminal**, go to the repository root, and run:

```bash
. ~/export-esp.sh
rustc --version                                     # 1.95.0 or newer
(cd firmware/esp32s3 && rustc --version)            # ends with (1.xx.0.0): the esp toolchain
command -v xtensa-esp32s3-elf-gcc                   # prints a path
espflash --version && esp-generate --version
groups | grep -o dialout                            # prints "dialout"
make check                                          # OK: Rust == Python
```

If every line gives the expected result, your computer is ready for Part 2.

Versions this guide was tested with:

| Tool | Version |
|---|---|
| rustc (stable) | 1.99.0 |
| esp toolchain | 1.97.0.0 |
| espup | 0.18.0 |
| espflash | 4.6.0 |
| esp-generate | 1.4.0 |
| uv | 0.11.11 |
| Python / numpy | 3.13.7 / 2.5.3 |

### Troubleshooting

**`cargo install` fails with "rustc 1.xx is not supported by the following packages"**
Your Rust is too old. Run `rustup update stable` and try again.

**`espflash` fails to build with "Package libudev was not found in the pkg-config search path"**
Install the missing system package with `sudo apt install libudev-dev`, then
re-run `cargo install espflash --locked`.

**Firmware build fails with "Xtensa linker `xtensa-esp32s3-elf-gcc` was not found in PATH"**
You opened a new terminal and didn't load the environment. Run `. ~/export-esp.sh`.

**Inside `firmware/esp32s3`, cargo complains that the `esp` toolchain is not installed**
Step 5 didn't complete. Run `espup install --targets esp32s3` again.

**`uv` prints "VIRTUAL_ENV=… does not match the project environment path `.venv`"**
Something in your shell (often pyenv or a leftover `activate`) exports
`VIRTUAL_ENV`. The warning is harmless: `uv` ignores that variable and uses the
project's `.venv`. To silence it, run `unset VIRTUAL_ENV` in that terminal.

**`make check` says "No such file or directory" for a `.tlm` or `.bin` file**
Step 7 was skipped or ran from the wrong folder. The model files must be inside
`models/`.

---

## Part 2 — Build the firmware

In this part you compile the firmware and confirm the resulting image is valid
and fits in the chip's memory, still without touching the board. If everything
here passes, any problem in Part 3 is about the board or the USB connection,
not the code.

### What's in `firmware/esp32s3`

The firmware follows the project template generated by `esp-generate` 1.4.0
(esp-hal 1.2), plus the inference code from `crates/tinyllm-core`.

| File | Role |
|---|---|
| `Cargo.toml` | ESP crate versions (esp-hal, esp-alloc, esp-println…) and the path dependency on `tinyllm-core` |
| `rust-toolchain.toml` | Selects the `esp` toolchain you installed in Part 1 |
| `.cargo/config.toml` | Build target, `build-std`, the flash runner, and **which model gets embedded** |
| `.cargo/esp-config.toml` | Runtime settings for esp-hal (log level) |
| `build.rs` | Adds the linker script and checks that the Xtensa linker is reachable |
| `src/main.rs` | Boots the chip, sets up the heap, runs the generation loop and prints to serial |

The model is not loaded at runtime from a file system. `include_bytes!` copies
`models/stories260K.tlm` and `models/tok512.bin` **into the firmware binary**
at compile time, so they end up in flash next to the code. The weights are read
directly from flash while the model runs, and are never copied to RAM.

### Step 1 — Compile

```bash
. ~/export-esp.sh                # if this terminal doesn't have it yet
cd firmware/esp32s3
cargo build --release
```

The first build takes about a minute: there is no prebuilt standard library
for the Xtensa target, so Cargo compiles `core` and `alloc` from source
(`build-std` in `.cargo/config.toml`). Later builds take a few seconds.

The build ends with:

```
warning: linker stderr: .../ld: .../tinyllm_esp32s3-... has a LOAD segment with RWX permissions
warning: `tinyllm-esp32s3` (bin "tinyllm-esp32s3") generated 1 warning
    Finished `release` profile [optimized + debuginfo] target(s) in ...
```

The RWX warning is expected: a fresh `esp-generate` project prints it too.

### Step 2 — Check the flash image

`espflash save-image` converts the compiled ELF into the exact binary that
gets written to flash, without needing a board:

```bash
espflash save-image --chip esp32s3 --flash-size 16mb \
    target/xtensa-esp32s3-none-elf/release/tinyllm-esp32s3 /tmp/tinyllm.bin
```

Expected:

```
Chip type:         esp32s3
...
App/part. size:    445,952/16,384,000 bytes, 2.72%
... Image successfully saved!
```

The firmware uses under 3% of the 16 MB flash, and most of its 440 KB is the
embedded model.

### Step 3 — Check where the memory goes

```bash
xtensa-esp32s3-elf-size -A target/xtensa-esp32s3-none-elf/release/tinyllm-esp32s3 \
    | grep -E '^\.(rodata|bss|stack|dram2_uninit) '
```

Expected (sizes in bytes):

```
.bss                    229484   ...
.rodata                 312916   ...
.stack                   95904   ...
.dram2_uninit            73744   ...
```

What each line means:

```
 FLASH (16 MB)
 └─ .rodata        ~305 KB   model weights (270 KB) + tokenizer + strings

 INTERNAL SRAM
 ├─ .bss           ~225 KB   main heap (224 KB): weight scales, activations, logits
 ├─ .stack          ~95 KB   whatever SRAM is left becomes the stack
 └─ .dram2_uninit    72 KB   RAM the bootloader used, reclaimed as a spare heap

 PSRAM (8 MB, external chip; not in the size output, set up at boot)
 └─ heap #3        8 MB      the KV cache, and whatever else doesn't fit in SRAM
```

The firmware reserves a KV cache for every position the model supports (512
tokens for stories260K), so a story can go on until the model itself ends it.
That cache is **two contiguous 320 KB blocks**, one for keys and one for
values. A block cannot be split between heap regions, and neither fits in
SRAM, so both land in PSRAM.

The allocator fills the heaps **in the order they are registered**: the main
SRAM heap first, then the reclaimed region, and the PSRAM last. SRAM is
faster, so the small buffers touched on every token (weight scales,
activations, logits, the tokenizer index) stay in SRAM, and only the big
blocks overflow into PSRAM. Moving the KV cache to PSRAM costs about 5% of
speed: a 128-token story ran at 33.9 tok/s with the cache in SRAM and 32.1
tok/s with it in PSRAM.

### Settings you can change

| Setting | Where | Default | Notes |
|---|---|---|---|
| Maximum story length | the model (`seq_len`) | 512 tokens for stories260K, 256 for stories15M | The KV cache is sized for the model's full length. Use `/steps` in the console to stop stories earlier |
| `DEFAULT_PROMPT`, `DEFAULT_TEMPERATURE` | `src/main.rs` | `"Once upon a time"`, `0.8` | Starting values only: you change both from the console without rebuilding (Part 4) |
| Embedded model and tokenizer | `make flash MODEL=… TOK=…` | `stories260K`, `tok512.bin` | Files in `models/`. Running `cargo` directly uses `TINYLLM_MODEL` / `TINYLLM_TOKENIZER` in `.cargo/config.toml` instead |
| Flash mode and clock | `runner` in `.cargo/config.toml` | `--flash-mode dio --flash-freq 80mhz` | See below |

Changes to these settings need a rebuild (Step 1), because they and the model
are compiled into the firmware. The flash settings are the exception:
`espflash` applies them when it writes the board, so they only need a
re-flash. Prompt, temperature, length and seed are runtime settings: Part 4
shows how to change them from the console.

**Why the flash clock matters.** The weights stay in flash and the model reads
all of them for every token, so the flash speed directly limits tokens per
second. Measured on stories260K with 128-token stories: 21.9 tok/s at
40 MHz, **33.5 tok/s at 80 MHz**. Do not switch to `qio`: `espflash` writes the bootloader in QIO mode
too, the chip's ROM can't enable quad mode on the flash chip, and the board
gets stuck in a reset loop (see Part 3, Troubleshooting).

### Updating esp-hal later

The esp-rs crates change quickly. To move to newer versions, generate a fresh
reference project **outside the repository** and compare it with
`firmware/esp32s3`:

```bash
cd /tmp
esp-generate --headless -o esp32s3 -o esp32s3-wroom-1-octal-psram \
    -o unstable-hal -o alloc -o log -o esp-backtrace ref-s3
```

Then compare `Cargo.toml` (versions and features), `.cargo/config.toml`,
`.cargo/esp-config.toml`, `build.rs`, and the start of `main()` in
`ref-s3/src/bin/main.rs` (chip init, heap setup, app descriptor). Keep this
project's `tinyllm-core` dependency, `opt-level = 3`, and the `TINYLLM_*`
variables.

**Keep the `esp-println` line as it is** (`default-features = false` with
`jtag-serial`), even though the template uses the defaults. In esp-println
0.18.0 the default `auto` mode reads the wrong register on the ESP32-S3: it
reads the USB receive FIFO instead of the interrupt status register, so every
`print!` swallows characters you type into the console. Only switch back to
the defaults after checking that a newer esp-println fixed it (the constant
`USB_DEVICE_INT_RAW` for `esp32s3` should be `0x60038008`, not `0x60038000`).

### Part 2 checklist

From `firmware/esp32s3`, with `~/export-esp.sh` loaded:

```bash
cargo build --release                                            # Finished, only the RWX warning
espflash save-image --chip esp32s3 --flash-size 16mb \
    target/xtensa-esp32s3-none-elf/release/tinyllm-esp32s3 /tmp/tinyllm.bin   # Image successfully saved!
```

If both pass, the firmware is ready for the board.

### Troubleshooting

**Build fails with "Xtensa linker `xtensa-esp32s3-elf-gcc` was not found in PATH or could not be executed"**
The `build.rs` check caught a terminal without the Xtensa environment. Run
`. ~/export-esp.sh` and build again.

**Build fails with "couldn't read `…/models/stories260K.tlm`"**
The model is embedded at compile time, so it must exist before you build. Run
`make models` from the repository root (Part 1, Step 7).

**You changed the model file but the firmware still runs the old one**
Cargo tracks files read by `include_bytes!` and rebuilds when they change. If
you switched models by editing `TINYLLM_MODEL`, run `cargo clean` once to be
sure.

---

## Part 3 — Flash and run on the board

### Step 1 — Connect the board

Use a USB-C cable that carries **data**. Many cables only charge, and with one
of those nothing below will work.

Some ESP32-S3 boards have one USB-C port and some have two, usually labeled
`USB` and `COM` (or `UART`). This guide uses the `USB` port, which talks
directly to the chip's built-in USB-Serial-JTAG controller. If your board only
has one port, that's the one.

Check that Linux sees the board:

```bash
ls /dev/ttyACM*          # expect /dev/ttyACM0
lsusb | grep 303a        # 303a is Espressif's USB vendor ID
```

The `lsusb` line tells you what state the board is in:

| `lsusb` shows | Meaning | What to do |
|---|---|---|
| `303a:1001 Espressif USB JTAG/serial debug unit` | The chip's own USB controller is active | Go to Step 3 |
| `303a:` with any other ID, e.g. `303a:4001 Espressif Device` | The firmware already on the board took over the USB port (common with factory demo firmware) | Do Step 2 once |

### Step 2 — First time only: enter download mode by hand

The firmware that ships on many boards drives the USB port itself, so
`espflash` cannot reset the chip into download mode. If you try,
`espflash board-info` fails with `Failed to connect to the device`.

Put the chip into download mode with the buttons:

1. **Hold** `BOOT` (sometimes labeled `IO0` or `B`).
2. While holding it, **press and release** `RST` (or `EN` / `R`).
3. **Release** `BOOT`.

The board disconnects and comes back as `303a:1001`. You only need to do this
once: the tinyllm firmware leaves the chip's USB controller in charge, so from
now on `espflash` resets the board on its own.

### Step 3 — Identify the board

```bash
espflash board-info
```

Expected:

```
Chip type:         esp32s3 (revision v0.2)
Crystal frequency: 40 MHz
Flash size:        16MB
Features:          WiFi, BLE, Embedded Flash
MAC address:       ...
```

**Flash size must say 16MB.** If it says 4MB or 8MB, your board is not an
N16R8 and the memory figures in this guide don't apply.

`espflash` doesn't report PSRAM. To confirm the 8 MB PSRAM, ask `esptool`:

```bash
uvx esptool flash-id
```

Look for `Embedded PSRAM 8MB (AP_3v3)` in the `Features` line. stories260K
doesn't need it, but the larger model in Part 5 does.

### Step 4 — Flash and run

From the repository root:

```bash
. ~/export-esp.sh        # if this terminal doesn't have it yet
make flash
```

`make flash` builds the firmware (Part 2), writes it to the board with the
flash clock at 80 MHz, resets the chip, and opens the serial monitor. The
bootloader messages confirm the flash settings:

```
I (29) boot.esp32s3: Boot SPI Speed : 80MHz
I (33) boot.esp32s3: SPI Mode       : DIO
```

Right after the bootloader, the firmware sets up the PSRAM and logs it:

```
INFO - 8388608 bytes of PSRAM
INFO - PSRAM initialized successfully in Octal SPI mode
INFO - PSRAM size: 8 MB
```

Then it waits **3 seconds**: the chip's USB disconnects on every reset, and
the pause gives your PC time to reconnect so the rest of the output isn't
lost. After that it loads the model and opens a console:

```
tinyllm-esp32s3
model: 269 KB in flash
dim=64 hidden=172 layers=5 heads=8 kv_heads=4 vocab=512 seq=512
state: 660 KB on the heap

Type the start of a story in English and press Enter.
Enter on an empty line repeats the last prompt.

  /temp <t>    0 = always the most likely word; 1.2+ = chaotic  [0.80]
  /steps <n>   total tokens, prompt included (1 to 512)         [512]
  /seed [n]    fixed seed; /seed alone = new one every story    [random]
  /verbose     show the tokens the model sees                   [off]
  /heap        memory usage
  /help        this help
Ctrl+R resets the board, Ctrl+C exits the monitor.

>
```

**Press Enter.** With nothing typed, the board uses the default prompt,
`Once upon a time`, and writes a story:

```
[seed …, temp 0.80, steps 512]
Once upon a time, …
-- … tokens in … ms (… tok/s), end of story
```

A story usually takes 10 to 25 seconds and ends on its own.

Each Enter picks a new random seed, so you get a different story every time.
To get exactly the story below, type `/seed 2026` and then press Enter:

```
[seed 2026, temp 0.80, steps 512]
Once upon a time, there was a little girl named Lily. She loved to play with her toys and feel happy. One day, she found a shiny piece of pictures in her pocket. She was so happy and climbed over.
But she wanted to share her owner. She started to climb a big tree. But it was too shiny and excited. The pictures could have a race that ever cleaned up. Lily looked up at the picture and bumped on a tree.
As they walked, Anna saw the picture of the picture. She thought it was a picture of a big plant. Suddenly, the waiter stopped. Lily's mom laughed and said, "That was a treasure. You can't blow it and make your mom shout."
Lily and her mom became good friends and played together. They played and made their feet even met the picture at the picnic because they were just like Tom feeling safe and careful.
-- 380 tokens in 17753 ms (21.40 tok/s), end of story
```

The model has only 260 thousand parameters, so expect stories that are
grammatical most of the time and logical only some of the time.

Monitor keys: **Ctrl+R** resets the board, **Ctrl+C** exits the monitor.

### Reading the output

- **`[seed …, temp …, steps …]`**: the settings used for this story. Write
  the seed down if you like a story: `/seed <that number>` brings it back.
- **`-- 380 tokens in 17753 ms (21.40 tok/s)`**: 380 new tokens in 17.8
  seconds. The time also includes processing the prompt. The speed depends on
  the story's length, because every new token looks back at all the previous
  ones: short stories run at about 35 tok/s, long ones at about 21. If short
  stories run well below 30 tok/s, the board was probably flashed at 40 MHz
  (see Troubleshooting).
- **`end of story`** means the model finished the story itself. It learned
  from its training data that every story is followed by a start-of-story
  token, so when it produces one, the current story is over.
  **`length limit`** means `/steps` cut the story off, often mid-sentence.

### Step 5 — Confirm the chip computes exactly what your PC computes

This is the end-to-end version of `make check` from Part 1. In the console,
type:

```
/temp 0
/steps 64
```

and press Enter on an empty line. The story must be **character-for-character
identical** to the one `make desktop` printed in Part 1:

```
[seed …, temp 0.00, steps 64]
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high. She as
-- 60 tokens in 1656 ms (36.23 tok/s), length limit
```

With temperature 0 the seed doesn't matter: the model always picks the most
likely next token. If the text matches, the board runs the exact same model
with the exact same arithmetic as your PC.

Afterwards, type `/temp 0.8` and `/steps 512` to go back to the defaults, or
press **Ctrl+R** to reset the board.

### Part 3 checklist

- `lsusb | grep 303a` shows `303a:1001`
- `espflash board-info` shows `esp32s3` and `Flash size: 16MB`
- `make flash` shows the console, and Enter prints a story and a `tok/s` line
- With `/temp 0` and `/steps 64`, the board's story matches `make desktop`

You now have a language model running on a microcontroller.

### Troubleshooting

**`espflash` fails with "Failed to connect to the device"**
The firmware on the board is holding the USB port. Do Step 2.

**After flashing, the monitor shows `boot:0x0 (DOWNLOAD(USB/UART0))` and `waiting for download`**
The chip went back to download mode instead of starting the firmware. This
can happen on the very first flash right after Step 2. Run `espflash monitor`
(or press `RST` once without `BOOT`) and it boots normally.

**The monitor shows nothing after the bootloader messages, or no `>` prompt**
The firmware printed the welcome text before the monitor reconnected. The
console is still waiting for you: type `/help` and press Enter, or press
**Ctrl+R**. If it keeps happening, increase the 3000 ms boot delay in
`src/main.rs`.

**Characters you type disappear, or commands arrive garbled (`/help` becomes `/l`)**
`esp-println` is running in its default `auto` mode, which has a bug on the
ESP32-S3 (see Part 2, "Updating esp-hal later"). Make sure
`firmware/esp32s3/Cargo.toml` has `default-features = false` and the
`jtag-serial` feature on `esp-println`, then run `make flash`.

**No output at all on the board's `COM` / `UART` port**
The firmware only prints on the native `USB` port (a consequence of the fix
above). Plug the cable into the `USB` port.

**Short stories run at about 22 tok/s instead of 35**
Compare with a short story (`/steps 64`), since long stories are slower anyway.
The board was written at the default 40 MHz flash clock, which happens if you
call `espflash flash` by hand. Use `make flash` (or `cargo run --release` in
`firmware/esp32s3`), which passes `--flash-freq 80mhz`. The bootloader line
`Boot SPI Speed` tells you which clock the board is using.

**The board resets in a loop, printing `mode:QIO` and `ets_loader.c 78`**
It was flashed with `--flash-mode qio`, and the chip can't read its own
bootloader in that mode. The board isn't damaged: run `make flash` again,
which uses DIO. Flashing still works during the loop because the chip's ROM
handles it.

**`Permission denied` on `/dev/ttyACM0`**
Your user is not in the `dialout` group yet, or you haven't logged out and back
in since adding it (Part 1, Step 2).

**`/dev/ttyACM0` doesn't exist**
Try another cable (it may be charge-only) or the board's other USB port. Check
`lsusb` for any `303a:` device.

---

## Part 4 — Play with the board

Everything in this part happens in the console from Part 3. Nothing needs a
rebuild.

### Opening the console

```bash
make flash               # builds, flashes, then opens the console
espflash monitor         # only opens the console (resets the board first)
```

`espflash monitor` doesn't need `~/export-esp.sh` and works from any folder.
Other serial programs (`screen`, `picocom`, Python's `pyserial`) work too, but
opening the port with them may reset the board: you'll see the welcome text
again.

### Console commands

| You type | What happens |
|---|---|
| any text + Enter | Generates a story that starts with your text |
| Enter alone | Generates again from the last prompt (with a new seed unless one is fixed) |
| `/temp <t>` | Sets the temperature, from `0` to `2` |
| `/steps <n>` | Sets the maximum length in tokens, prompt included, from 1 to the model's limit (512 for stories260K) |
| `/seed <n>` | Fixes the seed: the same prompt and settings always give the same story |
| `/seed` | Goes back to a new random seed for every story |
| `/verbose` | Toggles showing the tokens the model sees |
| `/heap` | Shows memory usage |
| `/help` | Shows the command list with the current values |

The console accepts plain ASCII. Backspace works; arrow keys are ignored.
Settings last until the board resets.

### Things to try

**Temperature** controls how adventurous the model is. It divides the
model's scores before turning them into probabilities:

| `/temp` | Behaviour |
|---|---|
| `0` | Always the single most likely token. Same text every time, often repetitive |
| `0.5` | Safe and coherent, little variety |
| `0.8` | The default: a balance between coherence and surprise |
| `1.2` and up | Unusual word choices, then invented words, then nonsense |

Fix a seed (`/seed 7`) and run the same prompt at different temperatures to
see the effect on one story. With `/steps 50` and `Tom had a big dog`:

```
1.2  Tom had a big dog named Max. He loved to play with his room before eating cake all day. One day, he tried to jump in
1.5  Tom had a big dog new wheat had like broken. The cup was big and hot on door. No long tummy's locky pl
2.0  Tom had a big dogd. Tom nutt and picture sack! Joe called happhesques and the sthee. He'm doing up in
```

**Prompts.** The model was trained only on short English stories for small
children, written with a vocabulary of about 1,500 simple words. It continues
prompts like these well:

```
Tom had a big dog
One day, Lily went to the park
The little bird was sad because
```

Prompts outside that world still produce text, but it quickly drifts back to
toys, parks and moms: `The stock market crashed` continues with *"in the sea.
Mom was very excited and kind. She looked all around the house."*

**See the tokens.** Type `/verbose`, then a prompt:

```
> Tim went to
[seed 7, temp 0.00, steps 40]
prompt = 5 tokens: 1:"\n<s>\n" 326:" Tim" 263:" w" 377:"ent" 267:" to"
Tim went to| the| p|ar|k| with| his| mom|.| They| saw| a| big| b|o|x|.| ...
```

The first line shows how your prompt was cut into tokens (`id:"text"`).
Token 1 is the invisible start-of-text marker. After that, each `|` marks
where one generated token ends and the next begins. The vocabulary has only
512 tokens, so common words are whole tokens (`" the"`, `" mom"`) and rarer
ones are spelled in pieces (`" p" "ar" "k"`).

### Experiment on your PC first

The desktop CLI from Part 1 runs the same code at thousands of tokens per
second, and with the same settings it produces **the same story** as the
board. Use it to explore quickly, then reproduce your favourite on the board:

```bash
cargo run -q --release -p tinyllm-desktop -- models/stories260K.tlm models/tok512.bin \
    --prompt "Tom had a big dog" --steps 40 --temp 0.9 --seed 7
```

prints the same text as typing `/temp 0.9`, `/steps 40`, `/seed 7` and
`Tom had a big dog` in the console:

```
Tom had a big dog named Max. He loved to play with his clay. Every day, he would untangle him to m
```

### Adding your own debug output

The console shows everything the firmware prints. To see more of what
happens inside, add prints to `firmware/esp32s3/src/main.rs` and run
`make flash`. Two kinds are available:

- `println!(...)`: always printed.
- `log::debug!(...)`, `log::info!(...)` and so on: printed only if their level
  is enabled by `ESP_LOG` in `firmware/esp32s3/.cargo/esp-config.toml`
  (default `"info"`). The level is read **at compile time**, so changing it
  needs `make flash` too.

For example, to time every token, change the start of the loop in
`generate()`:

```rust
for pos in 0..steps {
    let t = Instant::now();
    let logits = model.forward(state, token, pos);
    log::debug!("pos={} token={} {} us", pos, token, t.elapsed().as_micros());
```

and set `ESP_LOG="debug"`.

If the firmware crashes, `make flash` shows a backtrace with function names.
With `espflash monitor` alone, give it the firmware file so it can decode the
addresses:

```bash
espflash monitor --elf firmware/esp32s3/target/xtensa-esp32s3-none-elf/release/tinyllm-esp32s3
```

---

## Part 5 — Run a bigger model

stories260K is a good first model, but at 260 thousand parameters its
stories often lose the thread. **stories15M** has 15 million parameters, 60×
more, and writes noticeably more coherent text. It is also the largest of
Karpathy's tinyllamas that fits on this board.

| | stories260K | stories15M |
|---|---|---|
| Parameters | 0.26 M | 15.2 M |
| Vocabulary | 512 tokens | 32,000 tokens |
| `.tlm` file | 270 KB | 15.0 MB |
| Flash used by the firmware | 2.7% | **97.9%** |
| RAM used | ~700 KB: 45 KB in SRAM, the 640 KB KV cache in PSRAM | ~4.3 MB: SRAM full, 3.9 MB in PSRAM |
| Speed on the board | 20 to 35 tokens/s | **~0.7 tokens/s** |
| Time to flash | a few seconds | ~2 min 15 s |

It is about 40× slower because the board reads all 15 MB of weights from flash
for every single token.

### Step 1 — Get the files

`make models` in Part 1 already downloaded stories15M and converted it. Check
that both files are there:

```bash
ls -l models/stories15M.tlm models/tokenizer.bin
```

stories15M uses the standard 32k-token Llama 2 tokenizer, `tokenizer.bin`,
instead of `tok512.bin`. If either file is missing, run `make models` again.

Every `make` target accepts `MODEL=` and `TOK=`. Always pass the tokenizer
that belongs to the model: with the wrong one, the output is garbage or the
program crashes.

### Step 2 — Validate on your PC

```bash
make desktop MODEL=stories15M TOK=tokenizer.bin
```

Expected:

```
model: dim=288 hidden=768 layers=6 heads=6 kv_heads=6 vocab=32000 seq=256
state: 1001 KB on the heap
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was
60 tokens generated in 0.30s (199.1 tok/s)
```

`make check MODEL=stories15M TOK=tokenizer.bin` reports a difference, and that
is expected. Both implementations agree on every token up to *"It was the
sun!"*. At that point the two most likely next tokens (`" "` and `" She"`)
score within **0.0025** of each other, while the typical gap between the top
two is about 2.5. The tiny rounding differences between Rust and numpy decide
that near-tie in opposite directions, and the stories continue differently
from there. A bug would show up much earlier and at a point with a clear
winner.

### Step 3 — Flash and run

```bash
make flash MODEL=stories15M TOK=tokenizer.bin
```

Writing 16 MB over USB takes a bit over two minutes; let it finish. After the
PSRAM lines, the console shows the new model:

```
tinyllm-esp32s3
model: 15041 KB in flash
dim=288 hidden=768 layers=6 heads=6 kv_heads=6 vocab=32000 seq=256
state: 3598 KB on the heap
```

A full 256-token story takes more than five minutes, so start with a short
one. Type `/temp 0`, `/steps 20`, then press Enter on an empty line:

```
[seed …, temp 0.00, steps 20]
Once upon a time, there was a little girl named Lily. She loved to play outside in
-- 16 tokens in 22238 ms (0.71 tok/s), length limit
```

Words appear about one per second. It is the same text as on the PC, as in
Part 3.

`/heap` shows where the memory went:

```
Internal | ██████████████████████████████████░ | Used: 99% (Used 227512 of 229376, free: 1864)
Internal | ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ | Used: 8% (Used 6144 of 73744, free: 67600)
External | ████████████████░░░░░░░░░░░░░░░░░░░ | Used: 48% (Used 4050944 of 8388608, free: 4337664)
```

The SRAM heap filled up first, and everything else went to the PSRAM
(`External`), as described in Part 2.

### Going back to stories260K

```bash
make flash
```

Without `MODEL=`, the Makefile uses stories260K again. Cargo notices the change
and rebuilds on its own, and flashing takes seconds again.

### Will another model fit?

Any checkpoint in llama2.c's original format can be used. Put the `.bin` and
its tokenizer in `models/`, then pass their names to the same `make` targets:

```bash
make export  MODEL=<name>                    # models/<name>.bin -> models/<name>.tlm
make desktop MODEL=<name> TOK=<tokenizer>    # try it on the PC
make flash   MODEL=<name> TOK=<tokenizer>    # put it on the board
```

To run on this board, it has to pass four checks:

1. **Format:** `make export` finishes and prints a sensible `config:`.
   `truncated file` means the file isn't in llama2.c's original format.
2. **Correctness:** `make desktop` produces readable text with the matching
   tokenizer.
3. **Flash:** the `.tlm`, plus the tokenizer file, plus about 200 KB of
   firmware code must stay under 16,384,000 bytes. `espflash` refuses to write
   an image that doesn't fit. stories15M uses 97.9% of that; this rules out
   stories42M and stories110M.
4. **RAM:** the total from `/heap` must fit in the 224 KB + 72 KB of SRAM plus
   8 MB of PSRAM. On the PC, `make desktop … STEPS=<seq_len>` prints the state
   size (KV cache and activations); the firmware always reserves the model's
   full `seq_len`. On the board, add about 8 bytes per vocabulary
   entry for the logits and 12 for the tokenizer index, plus the per-row weight
   scales.

### Troubleshooting

**`make flash` seems stuck while writing**
A 16 MB image takes over two minutes to write. Wait for `Flashing has completed!`.

**The console seems frozen after pressing Enter**
stories15M is slow: about one token per second, and the first words of the
prompt also take a second each. Use `/steps 40` for quicker tests.

**`make check` fails with stories15M**
See Step 2: a late divergence after a near-tie is expected with this model.
