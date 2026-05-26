import itertools

# WRAPPER ######################################################################

class BatchedDataset:
    """Provide a length to datasets made from an iterator."""

    def __init__(self, dataset_obj: object, batch_dim: int, batch_num: int=None) -> None:
        self._dataset = dataset_obj
        self._batch = int(batch_dim)
        self._count = batch_num

    def __len__(self) -> int:
        return len(self._dataset) // max(1, (self._batch))

    def __iter__(self) -> object:
        return itertools.islice(
            self._dataset.iter(batch_size=self._batch),
            self._count)
