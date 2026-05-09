# WRAPPER ######################################################################

class BatchedDataset:
    """Provide a length to datasets made from an iterator."""

    def __init__(self, dataset_obj: object, batch_dim: int) -> None:
        self._dataset = dataset_obj
        self._batch = int(batch_dim)

    def __len__(self) -> int:
        return len(self._dataset) // max(1, (self._batch))

    def __iter__(self) -> object:
        return self._dataset.iter(batch_size=self._batch)
