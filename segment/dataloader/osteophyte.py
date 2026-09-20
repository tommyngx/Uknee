"""Osteophyte-only mask contract used by RWKV_UNetV6b."""

import numpy as np
from torch.utils.data import Dataset

OSTEOPHYTE_SOURCE_CLASSES = (
    (6, "Lat Fem Osteophyte"),
    (7, "Med Fem Osteophyte"),
    (8, "Lat Tib Osteophyte"),
    (9, "Med Tib Osteophyte"),
)
OSTEOPHYTE_CLASS_INFO = (
    {"class_id": 0, "name": "background", "source_class_id": 0},
    *(
        {"class_id": output_id, "name": name, "source_class_id": source_id}
        for output_id, (source_id, name) in enumerate(OSTEOPHYTE_SOURCE_CLASSES, start=1)
    ),
)


def remap_osteophyte_mask(mask):
    source = np.asarray(mask)
    remapped = np.zeros(source.shape, dtype=np.int64)
    for output_id, (source_id, _) in enumerate(OSTEOPHYTE_SOURCE_CLASSES, start=1):
        remapped[source == source_id] = output_id
    return remapped


class OsteophyteOnlyDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.class_info = [dict(item) for item in OSTEOPHYTE_CLASS_INFO]
        self.num_classes = len(self.class_info)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["label"] = remap_osteophyte_mask(sample["label"])
        return sample


__all__ = ["OSTEOPHYTE_CLASS_INFO", "OSTEOPHYTE_SOURCE_CLASSES",
           "OsteophyteOnlyDataset", "remap_osteophyte_mask"]
