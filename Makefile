MODEL ?= stories260K
TOK   ?= tok512.bin
PROMPT ?= Once upon a time
STEPS ?= 64
PY    ?= uv run python

# creates/updates .venv from uv.lock
setup:
	uv sync

export:
	$(PY) tools/export.py models/$(MODEL).bin models/$(MODEL).tlm

# downloads both models with their tokenizers and converts them to .tlm;
# only fetches or converts what is missing or out of date
models: models/stories260K.tlm models/tok512.bin models/stories15M.tlm models/tokenizer.bin

TINYLLAMAS = https://huggingface.co/karpathy/tinyllamas/resolve/main

# downloads go to a .part file first, so an interrupted transfer is retried
# instead of leaving a truncated file that make would consider up to date
models/stories260K.bin:
	curl -fL --progress-bar -o $@.part $(TINYLLAMAS)/stories260K/stories260K.bin && mv $@.part $@
models/tok512.bin:
	curl -fL --progress-bar -o $@.part $(TINYLLAMAS)/stories260K/tok512.bin && mv $@.part $@
models/stories15M.bin:
	curl -fL --progress-bar -o $@.part $(TINYLLAMAS)/stories15M.bin && mv $@.part $@
models/tokenizer.bin:
	curl -fL --progress-bar -o $@.part https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin && mv $@.part $@

models/%.tlm: models/%.bin tools/export.py
	$(PY) tools/export.py $< $@

desktop:
	cargo run --release -p tinyllm-desktop -- models/$(MODEL).tlm models/$(TOK) --prompt "$(PROMPT)" --steps $(STEPS) --temp 0

reference:
	$(PY) tools/reference.py models/$(MODEL).tlm models/$(TOK) --prompt "$(PROMPT)" --steps $(STEPS)

# both outputs must be identical
check:
	@$(MAKE) -s desktop 2>/dev/null > /tmp/rust.txt
	@$(MAKE) -s reference > /tmp/py.txt
	@diff /tmp/rust.txt /tmp/py.txt && echo "OK: Rust == Python"

# embeds models/$(MODEL).tlm and models/$(TOK) (overrides the defaults in
# firmware/esp32s3/.cargo/config.toml), flashes the board and opens the console
flash:
	cd firmware/esp32s3 && \
	TINYLLM_MODEL=$(CURDIR)/models/$(MODEL).tlm \
	TINYLLM_TOKENIZER=$(CURDIR)/models/$(TOK) \
	cargo run --release

.PHONY: setup export models desktop reference check flash
