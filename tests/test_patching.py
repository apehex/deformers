from scripts.patching import decode_tokens_reversible, encode_tokens_reversible

def test_reversible_patching_preserves_non_truncated_tokens() -> None:
    __tokens = ["hello", " world", "🙂"]
    __patch = encode_tokens_reversible(tokens=__tokens, max_length=32)
    assert __patch.truncated == [False, False, False]
    assert decode_tokens_reversible(__patch) == __tokens

def test_reversible_patching_restores_truncated_tokens_with_overflow() -> None:
    __tokens = ["0123456789ABCDE"]
    __patch = encode_tokens_reversible(tokens=__tokens, max_length=8)
    assert __patch.truncated == [True]
    assert __patch.lengths == [15]
    assert __patch.overflow_ids == [[56, 57, 65, 66, 67, 68, 69]]
    assert decode_tokens_reversible(__patch) == __tokens
