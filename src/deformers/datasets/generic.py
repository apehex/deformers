# WRAPPER ######################################################################

class BatchedDataset:
    """Provide a length to datasets made from an iterator."""

    def __init__(self, dataset_obj: object, batch_dim: int, batch_num: int=-1) -> None:
        self._dataset = dataset_obj
        self._batch = int(batch_dim)
        self._batch_num = int(batch_num)

    def __len__(self) -> int:
        __size = len(self._dataset) // max(1, self._batch)
        return __size if (self._batch_num < 1) else min(__size, self._batch_num)

    def __iter__(self) -> object:
        __iterator = self._dataset.iter(batch_size=self._batch)
        for __i, __row in enumerate(__iterator):
            if (self._batch_num > 0) and (__i >= self._batch_num):
                break
            yield __row
