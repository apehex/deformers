import os.path

import torch
import torch.nn

import deformers.layers.prefix

# META #########################################################################

SHAPE_MSG = 'Inputs must be rank {}, got shape={} with patch_dim={}.'
PATH_MSG = 'No model checkpoint found at {path}.'

# BYTE #########################################################################

class CompositeBytePrefix(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int, # dimension of each byte embedding
        patch_dim: int=-1, # split the sequence when patch is positive
        hidden_dim: int=-1, # defaults to the output dimension when -1
        output_dim: int=-1, # defaults to G*E (patch * embed) when -1
        vocab_dim: int=256, # number of # byte values
        padding_idx: int=128, # default padding value
        block_num: int=4, # number of transformer blocks
        head_num: int=4, # number of self-attention heads
        dropout_rate: float=0.0, # dropout for the attention weights
        **kwargs: dict,
    ) -> None:
        super(CompositeBytePrefix, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'embed_dim': int(embed_dim),
            'patch_dim': int(patch_dim),
            'hidden_dim': int(hidden_dim),
            'output_dim': int(output_dim),
            'vocab_dim': int(vocab_dim),
            'padding_idx': int(padding_idx),
            'block_num': int(block_num),
            'head_num': int(head_num),
            'dropout_rate': float(dropout_rate),}
        # submodule, initialized on first forward call
        self._embed = None
        self._blocks = None
        self._combine = None
        self._project = None
        self._built = False

    def build(
        self,
        shape: tuple,
        device: object=None,
        dtype: object=None,
    ) -> None:
        if not self._built:
            __shape = tuple(shape)
            # compute the default dimensions
            __embed_dim = self._config['embed_dim']
            __patch_dim = __shape[-1] if (self.config['patch_dim'] < 1) else self._config['patch_dim']
            __output_dim = __embed_dim * __patch_dim if (self._config['output_dim'] < 1) else self._config['output_dim']
            __hidden_dim = __output_dim if (self._config['hidden_dim'] < 1) else self._config['hidden_dim']
            # value + position embedding of each byte
            self._embed = deformers.layers.prefix.ByteEncoder(
                embed_dim=self._config['embed_dim'],
                patch_dim=self._config['patch_dim'],
                vocab_dim=self._config['vocab_dim'],
                padding_idx=self._config['padding_idx'])
            # model the interactions between the bytes, iteratively
            self._blocks = [
                deformers.layers.prefix.ByteTransformer(
                    head_num=self._config['head_num'],
                    dropout_rate=self._config['dropout_rate'])
                for _ in range(self._config['block_num'])]
            # combine all the byte vectors into a single token vector
            self._combine = deformers.layers.prefix.ByteMixer()
            # project into the latent space of the teacher model
            self._project = deformers.layers.prefix.TokenProjector(
                hidden_dim=__hidden_dim,
                output_dim=__output_dim)
            # create the weights according to the inputs' shape
            for __l in [self._embed] + self._blocks + [self._combine, self._project]:
                __l.build(shape=__shape, dtype=dtype, device=device)
                __shape = __l.output_shape(__shape)
            # register
            self._built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # the inputs are supposed to be integers, default to float32 weights
        self.build(shape=tuple(inputs.shape), dtype=torch.float32, device=inputs.device)
        # identify the positions with padding (B, T, G)
        __paddings = (inputs == self._config['padding_idx']).to(dtype=torch.bool, device=inputs.device)
        # embed the byte values and positions (B, T, G) => (B, T, G, E)
        __outputs = self._embed(inputs)
        # model the interactions between the bytes (B, T, G, E) => (B, T, G, E)
        for __i, __l in enumerate(self._blocks):
            # the first transformer block attends only to the non-padding bytes
            # subsequent blocks attend to all bytes, since all positions now have data
            __outputs = __l(inputs=__outputs, paddings=(__paddings if (__i == 0) else None))
        # combine into a single token vector and add a length embedding (B, T, G, E), (B, T, G) => (B, T, G*E)
        __outputs = self._combine(inputs=__outputs, paddings=__paddings)
        # project into the latent space of the teacher model (B, T, G*E) => (B, T, O)
        return self._project(__outputs)

    def output_shape(self, shape: tuple) -> tuple:
        __shape = tuple(shape)
        # (B, T, G) => (B, T, G, E) => (B, T, G*E) => (B, T, O)
        for __l in [self._embed] + self._blocks + [self._combine, self._project]:
            __shape = __l.output_shape(__shape)
        # (B, T, O)
        return __shape

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
        assert os.path.isfile(path), PATH_MSG.format(path)
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
