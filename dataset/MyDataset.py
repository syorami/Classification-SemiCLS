""" The Code is under Tencent Youtu Public Rule
"""
from PIL import Image
from torch.utils.data import Dataset


#val，l-train dataset
class MyDataset(Dataset):
    """
    Interface provided for customized data sets

    names_file：a txt file, each line in the form of "image_path label"

    transform: transform pipline for mydataset

    """
    def __init__(self, names_file, transform=None):
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.names_list[idx].split(' ')[0]
        image = Image.open(image_path)
        if(image.mode == 'L'):
            image = image.convert('RGB')
        label = int(self.names_list[idx].split(' ')[1])

        if self.transform:
            image = self.transform(image)

        return image, label


class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        """Get category ids of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.dataset.get_cat_ids(idx % self._ori_len)

    def get_ann_info(self, idx):
        """Get annotation of repeat dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.dataset.get_ann_info(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


