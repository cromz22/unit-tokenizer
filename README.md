# Unit Tokenizer

![pytest](https://github.com/cromz22/unit-bpe/actions/workflows/run_pytest.yml/badge.svg)

Tokenizers that operate on a sequence of integers.

## Requirements

- Python >= 3.9 (because of type hinting syntax)

## Features

- BPETokenizer
    - Byte-pair encoding algorithm

- RLETokenizer
    - Run-length encoding algorithm

- PackBitsTokenizer
    - Modified run-length encoding algorithm

## Installation for development

```
poetry install
pre-commit install
```

### Test

```
poetry run pytest
```

## Usage

See `tests/*.py`.
