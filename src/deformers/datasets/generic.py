import itertools

# WRAPPER ######################################################################

class BatchedDataset:
    """Provide a length to datasets made from an iterator."""

    def __init__(self, dataset_obj: object, batch_dim: int, batch_num: int=-1) -> None:
        self._dataset = dataset_obj
        self._batch = int(batch_dim)
        self._count = int(batch_num)

    def __len__(self) -> int:
        __size = len(self._dataset) // max(1, (self._batch))
        return __size if (self._count < 1) else min(__size, self._count)

    def __iter__(self) -> object:
        return itertools.islice(
            self._dataset.iter(batch_size=self._batch),
            None if (self._count < 1) else self._count)
