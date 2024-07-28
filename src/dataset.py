from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import dataclasses
from typing import List, Any


@dataclasses.dataclass
class GetDatasetSgm(Dataset):
    annotation: List[List[str]]
    class_values: List[int]
    augmentation: List[Any] = dataclasses.field(default_factory=list)

    def __getitem__(self, i):
        img_file, label_file = self.annotation[i]

        image = np.array(Image.open(img_file))
        mask = np.array(Image.open(label_file))

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # preprocess image for input
        image = image.transpose(2, 0, 1).astype('float32') / 255.

        masks = [(mask == v) for v in self.class_values]
        mask = np.array(masks, dtype='float32')

        return image, mask

    def __len__(self):
        return len(self.annotation)


def annotation_files(root_dir: str, img_dir: str, label_dir: str):
    # img_path: root_dir / case_dir / img_dir / img_file
    # label_path: root_dir / case_dir / label_dir / label_file

    anno_files = []

    for case_dir in Path(root_dir).iterdir():
        img_path = case_dir / img_dir
        label_path = case_dir / label_dir
        for img_file in img_path.iterdir():
            label_file = label_path / f"{img_file.stem}.png"
            anno_files.append([str(img_file), str(label_file)])

    return anno_files
