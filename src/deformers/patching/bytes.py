import torch

# PREPROCESSING ################################################################

def token_offsets_to_substrings(texts: list, offsets: list) -> list:
    """
    Convert tokenizer offset mappings to the actual substring for each token.

    The base tokenizer returns offset_mapping as a list of (start, end) pairs.
    Padding positions have offset (0, 0) and map to the empty string ''.

    Args:
        texts:   list of B source strings, one per sample.
        offsets: list of B lists of (start, end) pairs from the tokenizer
                 offset_mapping, including padding offsets.

    Returns:
        list of B lists of T substrings, one per token (empty for padding).
    """
    return [
        [__t[__s:__e] for (__s, __e) in __o]
        for (__t, __o) in zip(texts, offsets)]


def encode_token_substrings_to_bytes(
    tokens: list,
    byte_tokenizer,
    max_length: int=32,
    padding: str='max_length',
    truncation: bool=True,
) -> list:
    """
    Encode each token substring as a fixed-length sequence of byte ids.

    Each token string is independently encoded by the byte tokenizer.
    Shorter tokens are right-padded and longer ones are truncated so that
    every block has exactly max_length byte ids.

    Args:
        tokens:         list of B lists of T token substrings.
        byte_tokenizer: a ByteTokenizer instance.
        max_length:     fixed byte block size (default: 32, see docs/roadmap.md).
        padding:        padding strategy passed to byte_tokenizer.
        truncation:     whether to truncate to max_length.

    Returns:
        list of B lists of T lists of max_length byte ids (nested Python ints).
    """
    return [
        byte_tokenizer(
            __s,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            padding_side='right')['input_ids']
        for __s in tokens]


def decode_bytes_to_text(byte_ids: list, byte_tokenizer) -> list:
    """
    Decode a batch of byte-id sequences back to text strings.

    Padding byte ids (128) are automatically skipped by the byte tokenizer's
    convert_tokens_to_string, so no explicit stripping is needed.

    Args:
        byte_ids:       list of B lists of T lists of G byte ids.
        byte_tokenizer: a ByteTokenizer instance.

    Returns:
        list of B decoded text strings.
    """
    return [''.join(byte_tokenizer.decode(__s)) for __s in byte_ids]


def encode_texts(
    texts: list,
    tokenizer,
    byte_tokenizer,
    max_length: int=32,
    truncation: bool=True,
) -> dict:
    """
    Produce aligned (B, T, *) tensors from a batch of raw text strings.

    Uses the pretrained tokenizer to determine token boundaries (offsets) and
    then encodes each token substring as a fixed-size byte block.

    Assumptions:
    - Tokenizer boundaries are identical to the base model (qwen/qwen3.5-9b).
    - Byte block size defaults to L_max=32 bytes (see docs/roadmap.md).
    - Byte padding uses id=128 (as implemented by ByteTokenizer).

    Args:
        texts:          list of B raw text strings.
        tokenizer:      HF pretrained tokenizer (e.g. AutoTokenizer for Qwen).
        byte_tokenizer: ByteTokenizer instance.
        max_length:     fixed byte block size per token.
        truncation:     whether to truncate byte sequences to max_length.

    Returns:
        dict with keys:
            'input_ids':      LongTensor (B, T) - token ids from pretrained tokenizer.
            'attention_mask': LongTensor (B, T) - 1 for real tokens, 0 for padding.
            'byte_ids':       LongTensor (B, T, G) - byte ids per token block.
    """
    # tokenize with offset mapping so we can recover per-token substrings
    __encoding = tokenizer(
        texts,
        return_offsets_mapping=True,
        padding='longest',
        return_tensors='pt')
    # extract the offset pairs for the full batch
    __offsets = __encoding['offset_mapping'].tolist()
    # recover the original substring for each token in each sample
    __tokens = token_offsets_to_substrings(texts, __offsets)
    # encode each substring as a fixed-length byte block
    __byte_ids = encode_token_substrings_to_bytes(
        __tokens,
        byte_tokenizer=byte_tokenizer,
        max_length=max_length,
        truncation=truncation)
    return {
        'input_ids': __encoding['input_ids'],
        'attention_mask': __encoding['attention_mask'],
        'byte_ids': torch.tensor(__byte_ids, dtype=torch.long),}
