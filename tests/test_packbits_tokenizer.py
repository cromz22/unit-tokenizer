import os

from tokenizers import PackBitsTokenizer


def test_encode():
    tokenizer = PackBitsTokenizer()
    encoded = tokenizer._encode([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6])
    assert encoded == [4, 100, 2, 101, 6, 102, 0, 4, 103, 104, 105, 106]


def test_decode():
    tokenizer = PackBitsTokenizer()
    decoded = tokenizer._decode([4, 100, 2, 101, 6, 102, 0, 4, 103, 104, 105, 106])
    assert decoded == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6]


def test_batch_encode():
    tokenizer = PackBitsTokenizer()
    encoded = tokenizer.encode([[0, 0, 0, 0, 1, 1], [2, 2, 2, 2, 2, 2, 3, 4, 5, 6]])
    assert encoded == [[4, 100, 2, 101], [6, 102, 0, 4, 103, 104, 105, 106]]


def test_batch_decode():
    tokenizer = PackBitsTokenizer()
    decoded = tokenizer.decode([[4, 100, 2, 101], [6, 102, 0, 4, 103, 104, 105, 106]])
    assert decoded == [[0, 0, 0, 0, 1, 1], [2, 2, 2, 2, 2, 2, 3, 4, 5, 6]]
