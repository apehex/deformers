import math

import torch
import torch.nn

import mlable.layers.embedding
import mlable.layers.shaping
import mlable.layers.transformer

# EMBEDDING ####################################################################

class ByteEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int, # dimension of each byte embedding
        patch_dim: int=-1, # dimension of each byte group
        vocab_dim: int=256, # number of # byte values
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
            # the embedding of the padding is still trained to become scratchpad
            self._value = mlable.layers.embedding.CompositeEmbedding(
                input_dim=self._config['vocab_dim'],
                output_dim=self._config['embed_dim'],
                group_dim=self._config['patch_dim'],
                # padding_idx=self._config['padding_idx'],
                merge_axes=False)
            # byte position embedding
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
        self.build(shape=tuple(inputs.shape), dtype=torch.float32, device=inputs.device)
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
                elementwise_affine=True,
                dtype=dtype,
                device=device)
            # non causal self-attention, with padding masked out
            self._attend = mlable.layers.transformer.SelfAttention(
                head_num=self._config['head_num'],
                dropout_rate=self._config['dropout_rate'],
                attention_idx=-2,
                affine_opt=True)
            # pre-MLP norm
            self._norm1 = torch.nn.RMSNorm(
                normalized_shape=(int(shape[-1]),),
                elementwise_affine=True,
                dtype=dtype,
                device=device)
            # MLP gate
            self._gate = mlable.layers.transformer.GatedLinearUnit(
                hidden_dim=int(shape[-1]),
                output_dim=int(shape[-1]),
                affine_opt=False)
            # create all the weights, the same shape is kept throughout
            self._attend.build(shape=shape, dtype=dtype, device=device)
            self._gate.build(shape=shape, dtype=dtype, device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor=None) -> torch.Tensor:
        # lazy build, if necessary
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.device)
        # attention on the sequence of bytes (single token at once)
        __outputs = inputs + self._attend(inputs=self._norm0(inputs), paddings=paddings, is_causal=False)
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
                num_embeddings=__patch_dim + 1, # maximum length
                embedding_dim=__output_dim,
                dtype=dtype,
                device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor) -> torch.Tensor:
        # lazy build, if necessary
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.device)
        # invert the padding mask and make sure it is in the right dtype (B, T, G)
        __mask = (~paddings.to(dtype=torch.bool)).to(dtype=torch.long)
        # embed the length of each token, in bytes (B, T, G) => (B, T) => (B, T, G*E)
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

# TOKEN ########################################################################

class TokenProjector(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        **kwargs: dict,
    ) -> None:
        super(TokenProjector, self).__init__(**kwargs)
        # save for import, export, duplication etc
        self._config = {
            'hidden_dim': int(hidden_dim),
            'output_dim': int(output_dim),}
        # build at runtime
        self._norm = None
        self._project = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        # lazy build at runtime
        if not self._built:
            # normalize the composite embedding
            self._norm = torch.nn.RMSNorm(
                normalized_shape=(int(shape[-1]),),
                elementwise_affine=True,
                dtype=dtype,
                device=device)
            # project the composite embedding into the teacher's space
            self._project = mlable.layers.transformer.GatedLinearUnit(
                hidden_dim=self._config['hidden_dim'],
                output_dim=self._config['output_dim'],
                affine_opt=True)
            # create the weights according to the inputs' shape
            self._project.build(shape=shape, dtype=dtype, device=device)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # lazy build, if necessary
        self.build(shape=tuple(inputs.shape), dtype=inputs.dtype, device=inputs.device)
        # (B, T, G*E) => (B, T, H)
        return self._project(self._norm(inputs))

    def output_shape(self, shape: tuple) -> tuple:
        return self._project.output_shape(shape)

    def get_config(self) -> dict:
        return dict(self._config)

    @classmethod
    def from_config(cls, config: dict, **kwargs: dict) -> torch.nn.Module:
        return cls(**{**config, **kwargs})
