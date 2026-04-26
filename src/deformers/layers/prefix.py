import os.path
import math

import torch
import torch.nn

import mlable.layers.embedding
import mlable.layers.normalization
import mlable.layers.shaping
import mlable.layers.transformer

# META #########################################################################

SHAPE_MSG = 'Inputs must be rank {}, got shape={} with group_dim={}'

# EMBEDDING ####################################################################

class ByteEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int, # dimension of each byte embedding
        patch_dim: int=-1, # dimension of each byte group
        vocab_dim: int=256, # meant to embed bytes
        padding_idx: int=128, # default padding value
        **kwargs: dict,
    ) -> None:
        super(ByteEncoder, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'embed_dim': int(embed_dim),
            'patch_dim': int(patch_dim),
            'vocab_dim': int(vocab_dim),
            'padding_idx': int(padding_idx),}
        # build at runtime
        self._value = None
        self._position = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        # lazy build at runtime
        if not self._built:
            # divide only if necessary (B, T*G) => (B, T, G, 3) or (B, T, G) => (B, T, G, E)
            self._value = mlable.layers.embedding.CompositeEmbedding(
                input_dim=self._config['vocab_dim'],
                output_dim=self._config['embed_dim'],
                group_dim=self._config['patch_dim'],
                # padding_idx=self._config['padding_idx'],
                merge_axes=False)
            # byte position embedding (B, T, G, E) => (B, T, G, E)
            self._position = mlable.layers.embedding.PositionalEmbedding(
                input_axis=-2,
                output_axis=-1)
            # create all the weights according to the respective inputs' shape
            self._value.build(shape=shape, dtype=dtype, device=device)
            self._position.build(shape=self._value.output_shape(shape), dtype=dtype, device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # the inputs are supposed to be integers so default to float32
        self.build(shape=tuple(inputs.shape), dtype=torch.float32, device=inputs.dtype)
        # (B, T*G) => (B, T, G) => (B, T, G, E) => (B, T, G, E)
        return self._position(self._value(inputs.to(dtype=torch.long)))

    def output_shape(self, shape: tuple) -> tuple:
        return self._value.output_shape(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# TRANSFORMER ##################################################################

class ByteTransformer(torch.nn.Module):
    def __init__(
        self,
        head_num: int,
        dropout_rate: float=0.0,
        **kwargs: dict,
    ) -> None:
        super(ByteTransformer, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'head_num': int(head_num),
            'dropout_rate': float(dropout_rate),}
        # build at runtime
        self._norm0 = None
        self._attend = None
        self._norm1 = None
        self._gate = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        # lazy build at runtime
        if not self._built:
            # pre-attention norm
            self._norm0 = torch.nn.RMSNorm(
                normalized_shape=(int(shape[-1]),),
                elementwise_affine=True).to(dtype=dtype, device=device)
            # non causal self-attention, with padding masked out
            self._attend = mlable.layers.transformer.SelfAttention(
                head_num=self._config['head_num'],
                dropout_rate=self._config['dropout_rate'],
                attention_idx=-2,
                bias_opt=True)
            # pre-MLP norm
            self._norm1 = torch.nn.RMSNorm(
                normalized_shape=(int(shape[-1]),),
                elementwise_affine=True).to(dtype=dtype, device=device)
            # MLP gate
            self._gate = mlable.layers.transformer.GatedLinearUnit(
                hidden_dim=int(shape[-1]),
                output_dim=int(shape[-1]))
            # create all the weights, the same shape is kept throughout
            self._attend.build(shape=shape, dtype=dtype, device=device)
            self._gate.build(shape=shape, dtype=dtype, device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor=None, causal: bool=False) -> torch.Tensor:
        # lazy build, if necessary
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.dtype)
        # attention on the sequence of bytes (single token at once)
        __outputs = inputs + self._attend(inputs=self._norm0(inputs), paddings=paddings, is_causal=causal)
        # select the relevant data with the gate
        return __outputs + self._gate(self._norm1(__outputs))

    def output_shape(self, shape: tuple) -> tuple:
        return tuple(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# MIXING #######################################################################

class ByteMixer(torch.nn.Module):
    def __init__(
        self,
        **kwargs: dict,
    ) -> None:
        super(ByteMixer, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {}
        # build at runtime
        self._flatten = None
        self._measure = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        # lazy build at runtime
        if not self._built:
            # parse the input shape
            __patch_dim = int(shape[-2])
            __output_dim = math.prod(tuple(shape[-2:]))
            # flatten the sequence of byte vectors (no weights)
            self._flatten = mlable.layers.shaping.Merge(
                axis=-1,
                right=False)
            # encode the length of each token, in bytes
            self._measure = torch.nn.Embedding(
                num_embeddings=__patch_dim, # maximum length
                embedding_dim=__output_dim).to(dtype=dtype, device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor) -> torch.Tensor:
        # lazy build, if necessary
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.dtype)
        # invert the padding mask and make sure it is in the right dtype (B, T, G)
        __mask = (~paddings.to(dtype=torch.bool)).to(dtype=torch.long)
        # embed the length of each token, in bytes (B, T, G) => (B, T, G*E)
        __outputs = self._measure(__mask.sum(dim=-1))
        # merge all the byte vectors (B, T, G, E) => (B, T, G*E)
        return __outputs + self._flatten(inputs)

    def output_shape(self, shape: tuple) -> tuple:
        return self._flatten.output_shape(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

# BYTE #########################################################################

class CompositeBytePrefix(torch.nn.Module):
    """
    Alternative prefix: composite byte embeddings projected to the hidden
    space of the base language model.

    Assumptions:
    - Base model is qwen/qwen3.5-9b with hidden_size=4096.
    - Tokenizer boundaries are identical to the base model.
    - Trunk and lm_head are frozen; only this module is trainable.
    - Byte block size defaults to L_max=32 (see docs/roadmap.md).
    - The byte tokenizer uses pad_id=128 (as in ByteTokenizer).

    Inputs:
    - group_dim <= 0: inputs must be rank-3 (B, T, G) already split into byte
      blocks; CompositeEmbedding is called with group_dim=-1.
    - group_dim > 0: inputs may be rank-2 (B, T*G) flat; CompositeEmbedding
      splits them by group_dim internally.

    Outputs:
    Always (B, T, H) float.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_dim: int=256,
        latent_dim: int=-1,
        group_dim: int=-1,
        norm_opt: bool=True,
        attn_opt: bool=True,
        **kwargs: dict,
    ) -> None:
        super(CompositeBytePrefix, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'embed_dim': int(embed_dim),
            'vocab_dim': int(vocab_dim),
            'latent_dim': int(latent_dim),
            'group_dim': int(group_dim),
            'norm_opt': bool(norm_opt),
            'attn_opt': bool(attn_opt),}
        # submodule, initialized on first forward call
        self._layers = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        if not self._built:
            __layers = []
            # actual group dimension: last dim of input when not configured
            __group_dim = shape[-1] if (self._config['group_dim'] < 1) else self._config['group_dim']
            # merged byte embedding dimension after CompositeEmbedding
            __embed_dim = __group_dim * self._config['embed_dim']
            # projection target dimension: defaults to merged embed dim
            __latent_dim = __embed_dim if (self._config['latent_dim'] < 1) else self._config['latent_dim']
            # divide only if necessary (B, T*G) => (B, T, G) or (B, T, G) => (B, T, G)
            __layers.append(mlable.layers.shaping.Divide(
                axis=-1,
                factor=max(1, self._config['group_dim']),
                insert=bool(self._config['group_dim'] > 1),
                right=bool(self._config['group_dim'] > 1)))
            # (B, T, G) => (B, T, G, E)
            __layers.append(torch.nn.Embedding(
                num_embeddings=self._config['vocab_dim'],
                embedding_dim=self._config['embed_dim']))
            # (B, T, G, E) => (B, T, G, E)
            if self._config['attn_opt']:
                __layers.append(torch.nn.MultiheadAttention(
                    embed_dim=self._config['embed_dim'],
                    num_heads=4,
                    batch_first=True,
                    bias=True))
            # (B, T, G, E) => (B, T, G*E)
            __layers.append(mlable.layers.shaping.Merge(
                axis=-1,
                right=False))
            # (B, T, G*E) => (B, T, G*E)
            if self._config['norm_opt']:
                __layers.append(torch.nn.RMSNorm(
                    normalized_shape=(__embed_dim,),
                    elementwise_affine=True))
            # (B, T, G*E) => (B, T, G*E)
            __layers.append(torch.nn.Linear(
                in_features=__embed_dim,
                out_features=__embed_dim,
                bias=True))
            # (B, T, G*E) => (B, T, G*E)
            __layers.append(torch.nn.SiLU())
            # (B, T, G*E) => (B, T, H)
            __layers.append(torch.nn.Linear(
                in_features=__embed_dim,
                out_features=__latent_dim,
                bias=True))
            # chain together the layers
            self._layers = torch.nn.Sequential(*__layers)
            # move to the target device at build time (no-op if device is None)
            self._layers = self._layers.to(device=device, dtype=dtype)
            # register the build
            self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        __shape = tuple(inputs.shape)
        __group = self._config.get('group_dim', -1)
        # unspecified group dimension: the inputs must already be split into blocks
        if __group <= 0:
            assert len(__shape) == 3,SHAPE_MSG.format(3, __shape, __group)
        # explicit group dimension: the inputs are expected to bea batch of flat sequences
        else:
            assert len(__shape) == 2, SHAPE_MSG.format(2, __shape, __group)
        # lazy init: build sub-layers on first call, placed on input device
        self.build(shape=__shape, device=inputs.device, dtype=None)
        # (B, T, G) => (B, T, H) or (B, T*G) => (B, T, H)
        return self._layers(inputs.to(dtype=torch.long))

    def output_shape(self, shape: tuple) -> tuple:
        # shape after embedding (B, T, G*E)
        __shape = self._layers[0].output_shape(shape)
        # default to G*E when H is not specified
        __dim = __shape[-1] if (self._config['latent_dim'] < 1) else self._config['latent_dim']
        # update the last dimension
        return __shape[:-1] + (__dim,)

    def get_config(self) -> dict:
        return dict(self._config)

    def save_checkpoint(self, path: str) -> None:
        torch.save({'config': self._config, 'state': self.state_dict()}, path)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        shape: tuple,
        device: object=None,
        **kwargs: dict
    ) -> torch.nn.Module:
        # check the disk
        assert os.path.isfile(path), f'model checkpoint not found: {path}'
        # parse the data
        __ckpt = torch.load(path, map_location=device, weights_only=True)
        # instantiate the model
        __prefix = cls.from_config(config=__ckpt['config'], **kwargs)
        # create the layers
        __prefix.build(shape=shape, device=None, dtype=None)
        # load the weights
        __prefix.load_state_dict(__ckpt['state'])
        # alternative transformer prefix
        return __prefix.to(device=device)
