import deformers.pipelines.patch

# PREPROCESSING ################################################################

def tensors_from_strings(
    text_arr: list[str],
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
) -> tuple[torch.Tensor]:
    # common casting arguments
    __args = {'dtype': dtype_obj, 'device': device_str,}
    # input_ids (B, T) and attention_mask (B, T)
    __inputs = text_tok(
        text_arr,
        return_offsets_mapping=True,
        max_length=sequence_dim,
        truncation='longest_first',
        padding='max_length')
    # byte patches (B, T, G)
    __encoded = deformers.pipelines.patch.tokenize_into_bytes(
        texts_arr=text_arr,
        offsets_arr=__inputs['offset_mapping'],
        patch_dim=patch_dim,
        tokenizer_obj=byte_tok)
    # format as tensors
    __mask_arr = torch.tensor(__inputs['attention_mask'], **__args)
    __indices_arr = torch.tensor(__inputs['input_ids'], **__args)
    __bytes_arr = torch.tensor(__encoded, **__args)
    # (B, T), (B, T), (B, T, G)
    return __mask_arr, __indices_arr, __bytes_arr

def tensors_from_indices(
    indices_arr: list[list[int]],
    text_tok: object,
    byte_tok: object,
    sequence_dim: int,
    patch_dim: int,
    dtype_obj: object=torch.long,
    device_str: str='cpu',
) -> tuple[torch.Tensor]:
    # common casting arguments
    __args = {'dtype': dtype_obj, 'device': device_str,}
    # mapping {id => token} over the vocabulary
    __padding = text_tok.pad_token
    __mapping = {__i: ('' if __t == __padding else __t) for (__t, __i) in text_tok.get_vocab().items()}
    # input_ids (B, T) and attention_mask (B, T)
    __inputs = text_tok.pad(
        {'input_ids': indices_arr},
        max_length=sequence_dim,
        padding='max_length')
    # translaste the IDs into tokens
    __tokens = [[__mapping[__i] for __i in __r] for __r in __inputs['input_ids']]
    # byte patches (B, T, G)
    __encoded = deformers.pipelines.patch.tokenize_into_bytes(
        tokens_arr=__tokens,
        patch_dim=patch_dim,
        tokenizer_obj=byte_tok)
    # format as tensors
    __mask_arr = torch.tensor(__inputs['attention_mask'], **__args)
    __indices_arr = torch.tensor(__inputs['input_ids'], **__args)
    __bytes_arr = torch.tensor(__encoded, **__args)
    # (B, T), (B, T), (B, T, G)
    return __mask_arr, __indices_arr, __bytes_arr