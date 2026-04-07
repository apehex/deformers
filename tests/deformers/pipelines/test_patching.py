import pytest

import deformers.pipelines.patching as patching
import deformers.tokenizers.byte

# FIXTURES #####################################################################

@pytest.fixture
def tokenizer():
    return deformers.tokenizers.byte.ByteTokenizer(encoding='utf-8')

# PREPROCESSING ################################################################

def test_partition_handles_padding_offsets() -> None:
    __texts = ['hello world', 'abc']
    __offsets = [
        [(0, 5), (5, 11), (0, 0)],
        [(0, 3), (0, 0), (0, 0)],]
    # list[list[str]] with shape (B, T)
    __outputs = patching.partition_into_tokens(texts_arr=__texts, offsets_arr=__offsets)
    assert __outputs == [
        ['hello', ' world', ''],
        ['abc', '', ''],]

def test_encode_returns_fixed_patch_dim(tokenizer) -> None:
    __tokens = [['hi', '', ''], ['a', 'xyz', '01234']]
    __patch = 8
    # list[list[list[int]]] with shape (B, T, G)
    __outputs = patching.encode_into_bytes(
        tokens_arr=__tokens,
        patch_dim=__patch,
        tokenizer_obj=tokenizer)
    # test the batch axis
    assert isinstance(__outputs, list)
    assert len(__outputs) == 2
    # test the sequence axis
    for __sample in __outputs:
        assert isinstance(__sample, list)
        assert len(__sample) == 3
        # test the patch axis
        for __block in __sample:
            assert isinstance(__block, list)
            assert len(__block) == __patch
            assert all(isinstance(__x, int) for __x in __block)

def test_tokenize_into_bytes_integration_shape(tokenizer) -> None:
    __texts = ['hello world', 'tiny']
    __offsets = [
        [(0, 5), (5, 11), (0, 0)],
        [(0, 4), (0, 0), (0, 0)],]
    __patch = 6
    # list[list[list[int]]] with shape (B, T, G)
    __outputs = patching.tokenize_into_bytes(
        texts_arr=__texts,
        offsets_arr=__offsets,
        patch_dim=__patch,
        tokenizer_obj=tokenizer,)
    # test the batch axis
    assert isinstance(__outputs, list)
    assert len(__outputs) == 2
    # test the sequence axis
    for __sample in __outputs:
        assert isinstance(__sample, list)
        assert len(__sample) == 3
        # test the patch axis
        for __block in __sample:
            assert isinstance(__block, list)
            assert len(__block) == __patch
            assert all(isinstance(__x, int) for __x in __block)

# POSTPROCESSING ###############################################################

def test_decode_into_text_decodes_per_sample(tokenizer) -> None:
    # (B, T, G)
    __encoded = [
        tokenizer(['ab', 'cde', 'f'], max_length=4, truncation='longest_first', padding='max_length', padding_side='right')['input_ids'],
        tokenizer(['Z', '', ''], max_length=4, truncation='longest_first', padding='max_length', padding_side='right')['input_ids'],]
    # (B,)
    __outputs = patching.decode_into_text(bytes_arr=__encoded, tokenizer_obj=tokenizer)
    assert __outputs == ['abcdef', 'Z']
