import deformers.datasets.generic as _generic


class _Dataset:
    def __init__(self, size: int) -> None:
        self._rows = list(range(size))

    def __len__(self) -> int:
        return len(self._rows)

    def iter(self, batch_size: int) -> object:
        for __i in range(0, len(self._rows), batch_size):
            yield self._rows[__i:__i + batch_size]


class TestBatchedDataset:

    def test_len_uses_dataset_size_and_batch_dim(self):
        __dataset = _generic.BatchedDataset(dataset_obj=_Dataset(10), batch_dim=4)
        assert len(__dataset) == 2

    def test_len_honors_optional_batch_limit(self):
        __dataset = _generic.BatchedDataset(dataset_obj=_Dataset(20), batch_dim=4, batch_num=3)
        assert len(__dataset) == 3

    def test_iter_honors_optional_batch_limit(self):
        __dataset = _generic.BatchedDataset(dataset_obj=_Dataset(20), batch_dim=4, batch_num=2)
        assert list(iter(__dataset)) == [[0, 1, 2, 3], [4, 5, 6, 7]]
