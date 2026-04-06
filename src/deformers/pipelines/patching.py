import torch

import deformers.tokenizers.byte

# PREPROCESSING ################################################################

def partition_into_tokens(
    texts_arr: list[str],
    offsets_arr: list[list[tuple[int, int]]],
) -> list[list[str]]:
    """Map tokenizer offsets back to token substrings (padding -> '')."""
    return [
        [__t[__s:__e] for (__s, __e) in __o]
        for (__t, __o) in zip(texts_arr, offsets_arr)]

def encode_into_bytes(
    tokens_arr: list[list[str]],
    patch_dim: int=32, # enough for 97.4% of the tokens of qwen3.5
    tokenizer_obj: object=deformers.tokenizers.byte.ByteTokenizer(encoding='utf-8'),
) -> list[list[list[int]]]:
    """Encode each token substring as a fixed-length sequence of byte ids."""
    return [
        tokenizer_obj(
            __s,
            max_length=patch_dim,
            truncation='longest_first',
            padding='max_length',
            padding_side='right')['input_ids']
        for __s in tokens_arr]

def tokenize_into_bytes(
    texts_arr: list[str],
    text_tokenizer_obj: object,
    byte_tokenizer_obj: object=deformers.tokenizers.byte.ByteTokenizer(encoding='utf-8'),
    patch_dim: int=32,
) -> list[list[list[int]]]:
    """Produce aligned (B, T, G) tensors from a batch of raw text strings."""
    # tokenize with offset mapping so we can recover per-token substrings
    __encoding = text_tokenizer_obj(
        texts_arr,
        return_offsets_mapping=True,
        padding='longest')
    # recover the original substring for each token in each sample
    __tokens = partition_into_tokens(
        texts_arr=texts_arr,
        offsets_arr=__encoding['offset_mapping'])
    # encode each substring as a fixed-length byte block
    __outputs = encode_into_bytes(
        tokens_arr=__tokens,
        patch_dim=patch_dim,
        tokenizer_obj=byte_tokenizer_obj)

# POSTPROCESSING ###############################################################

def decode_into_text(
    bytes_arr: list[list[int]],
    tokenizer_obj: object=deformers.tokenizers.byte.ByteTokenizer(encoding='utf-8'),
) -> list[str]:
    """Decode a batch of byte sequences back to text strings."""
    return [''.join(tokenizer_obj.decode(__s)) for __s in bytes_arr]
