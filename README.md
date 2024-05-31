# Unit BPE

![pytest](https://github.com/cromz22/unit-bpe/actions/workflows/run_pytest.yml/badge.svg)

Plain BPE tokenizer that operates on a sequence of integers.

## Requirements

- Python >= 3.9 (because of type hinting syntax)

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

See `test_bpe_tokenizer.py`.
