PYTHON ?= python3
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
BIN := .venv/bin

FACE ?= input/face.png
AUDIO ?= input/audio.m4a
OUT ?= output/result.mp4

.PHONY: setup run test

setup: $(BIN)/lipsync
	$(BIN)/lipsync fetch

$(BIN)/lipsync: pyproject.toml
	$(PYTHON) -m venv .venv
	$(BIN)/pip install --extra-index-url $(TORCH_INDEX) -e '.[dev]'
	touch $@

run: setup
	$(BIN)/lipsync render $(FACE) $(AUDIO) -o $(OUT)
	$(BIN)/lipsync verify $(OUT) --face $(FACE) --audio $(AUDIO)

test: setup
	$(BIN)/pytest
