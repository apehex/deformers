import datasets
import torch

# UNIFORM ######################################################################

def _uniform_generator(
    dataset_len: int,
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int,
    seed_num: int=1337,
) -> iter:
    __gen = torch.Generator()
    __gen.manual_seed(seed_num)
    # number of samples
    for _ in range(dataset_len):
        # dictionary of columns
        yield {
            'indices': torch.randint(
                low=0,
                high=vocab_dim,
                size=(batch_dim, sequence_dim),
                generator=__gen),}

def build_uniform_dataset(
    dataset_len: int,
    vocab_dim: int,
    batch_dim: int,
    sequence_dim: int,
    seed_num: int=1337,
) -> datasets.Dataset:
    return datasets.Dataset.from_generator(
        generator=_uniform_generator,
        gen_kwargs={
            'dataset_len': dataset_len,
            'vocab_dim': vocab_dim,
            'batch_dim': batch_dim,
            'sequence_dim': sequence_dim,
            'seed_num': seed_num,})
