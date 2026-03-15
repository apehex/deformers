import copy
import json
import os
import transformers

class ByteTokenizer(transformers.PreTrainedTokenizer):
    """
    Special tokenizer that encodes text as a sequence of byte indexes.

    It behaves like a regular tokenizer, with a vocabulary of 256 (1 byte, 8 bits).
    Most of the logic is inherited from `PreTrainedTokenizer`.

    The special tokens are set to obsolete / unused Unicode codepoints.
    These codepoints have legacy meanings that can be leveraged.

    UTF-8 encoding is the most compact encoding scheme, but the number of
    bytes to encode a given character can very between 1 and 4.

    UTF-32-BE encoding is very sparse since it systematically represents
    characters with 4 bytes, most of which are 0.
    However, the fixed size allows to patch the input without overlap.

    The vocabulary is set for compatibility with the parent classes.
    In UTF-32-BE "a" is '\u0061' or `[0, 0, 0, 97]`.
    The vocabulary is used to convert the indexes back and forth but the
    actual character represented comes from the combination of indexes.

    Args:
        encoding (`str, defaults to 'utf-8'):
            The text to integer mapping used.
            'utf-8' is the popular choice, but for ML purposes 'utf-32-be' is recommanded.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0002'):
            A special token representing the beginning of a sentence.
            Defaults to '\u0002', which the unicode codepoint for "start of text".
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0003'):
            A special token representing the end of a sentence.
            Defaults to '\u0003', which the unicode codepoint for "end of text".
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0000'):
            A special token representing an out-of-vocabulary token.
            Defaults to '\u0000', which the unicode codepoint for "null".
        sep_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u001e'):
            A special token separating two different sentences in the same input.
            Defaults to '\u001e', which the unicode codepoint for "record separator".
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0080'):
            A special token used to make arrays of tokens the same size for batching purpose.
            Will then be ignored by attention mechanisms or loss computation.
            Defaults to '\u0080', which the unicode codepoint for "padding character".
        cls_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u001d'):
            A special token representing the class of the input (used by BERT for instance).
            Defaults to '\u001d', which the unicode codepoint for "group separator".
        mask_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u001a'):
            A special token representing a masked token.
            Defaults to '\u001a', which the unicode codepoint for "substitute".
    """

    def __init__(
        self,
        encoding: str='utf-8', # use utf-32-be for fixed patching
        bos_token: str='\u0002', # unicode "start of text"
        eos_token: str='\u0003', # unicode "end of text"
        unk_token: str='\u0000', # unicode "null"
        sep_token: str='\u001e', # unicode "record separator"
        pad_token: str='\u0080', # unicode "padding character"
        cls_token: str='\u001d', # unicode "group separator"
        mask_token: str='\u001a', # unicode "substitute"
        **kwargs,
    ) -> None:
        __kwargs = copy.deepcopy(kwargs)
        # save the encoding scheme
        self._encoding = encoding
        # enforce defaults
        __kwargs['additional_special_tokens'] = None # use the built-in special characters from Unicode
        __kwargs['split_special_tokens'] = True # in UTF-32, split the special codepoints into 4 bytes too
        # init the parent class
        super(ByteTokenizer, self).__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **__kwargs)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Each Unicode character is represented by a sequence of 1 to 4 tokens.
        These tokens are actually bytes converted to strings.

        The number of bytes required for each character depends on the Unicode scheme.
        The popular choice is 'utf-8', while 'utf-32-be' allows fixed size patching.
        """
        return list(chr(__b) for __b in text.encode(self._encoding))

    def _convert_token_to_id(self, token: str) -> int:
        """
        Interpret a single token as a string representation of a byte.
        """
        return ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Cast a single byte to its string representation.
        """
        return chr(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Each Unicode character is represented by a sequence of 1 to 4 tokens.
        These tokens are actually bytes converted to strings.

        This functions joins all the tokens to form the byte representation of
        the origin string. This byte sequence is then decoded back to text.
        """
        return bytes(ord(__c) for __c in tokens).decode(self._encoding, errors="ignore")

    @property
    def vocab_size(self) -> int:
        """
        The byte tokenizer does not use a vocabulary at all.
        It is implemented for compatibility with the parent classes.

        See `PreTrainedTokenizer.vocab_size`.
        """
        return 256

    def get_vocab(self) -> Dict[str, int]:
        """
        The byte tokenizer does not use a vocabulary at all.
        It is implemented for compatibility with the parent classes.

        See `PreTrainedTokenizerBase.get_vocab`.
        """
        return {chr(__i): __i for __i in range(256)}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        The byte tokenizer does not use a vocabulary at all.
        It is implemented for compatibility with the parent classes.

        See `PreTrainedTokenizerBase.save_vocabulary`.
        """
        __prefix = filename_prefix + '-' if filename_prefix else ''
        __path = os.path.join(save_directory, f'{__prefix}vocab.json')
        with open(__path, "w") as __file:
            json.dump(self.get_vocab(), __file)
        return (__path,)
