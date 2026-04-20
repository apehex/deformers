import datasets
import random

# UNIFORM ######################################################################

def _uniform_generator(
    dataset_len: int,
    vocab_dim: int,
    sequence_dim: int,
    seed_num: int=1337,
) -> iter:
    __gen = random.Random(seed_num)
    # number of samples
    for _ in range(dataset_len):
        # dictionary of columns
        yield {'indices': [__gen.randrange(vocab_dim) for _ in range(sequence_dim)],}

def build_uniform_dataset(
    dataset_len: int,
    vocab_dim: int,
    sequence_dim: int,
    seed_num: int=1337,
) -> datasets.Dataset:
    return datasets.Dataset.from_generator(
        generator=_uniform_generator,
        gen_kwargs={
            'dataset_len': dataset_len,
            'vocab_dim': vocab_dim,
            'sequence_dim': sequence_dim,
            'seed_num': seed_num,})
