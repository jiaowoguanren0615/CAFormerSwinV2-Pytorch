import torch
from PIL import Image
import glob
import os

import torchvision
torchvision.disable_beta_transforms_warning() # silence warning


import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset

from random import sample
import albumentations as A



def preprocess(x):
    return x / 255


def separate_class(x):
    first_dim = torch.where(x == 0, torch.ones_like(x), torch.zeros_like(x))
    second_dim = torch.where(x == 1, torch.ones_like(x), torch.zeros_like(x))
    #     third_dim = torch.where(x == 2, torch.ones_like(x), torch.zeros_like(x))
    return torch.cat((first_dim, second_dim)) / 1


def build_transform(img_size):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((img_size, img_size)),
        preprocess,
    ])

    target_transform = transforms.Compose([
        transforms.Resize((img_size // 4, img_size // 4)),
        separate_class
    ])

    trans = A.Compose([
        A.CenterCrop(img_size, img_size),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(),
        A.RandomBrightnessContrast(),
        A.OneOf([A.GridDistortion(p=0.5)], p=0.8)
    ])
    return transform, target_transform, trans



class KvasirDataSet(Dataset):
    def __init__(self, Kvasir_folder, ClinicDB_folder, img_size=256, train_mode=False):

        super(KvasirDataSet, self).__init__()
        self.img_size = img_size
        self.train_mode = train_mode

        self.img_files1 = glob.glob(os.path.join(Kvasir_folder, 'images', '*.jpg'))
        self.img_files2 = glob.glob(os.path.join(ClinicDB_folder, 'Original', '*.png'))

        if self.train_mode:
            self.img_files1 = self.img_files1
            self.img_files2 = self.img_files2

        else:  # use random 20% dataset for valid_data
            self.img_files1 = sample(self.img_files1, len(self.img_files1) // 5)
            self.img_files2 = sample(self.img_files2, len(self.img_files2) // 5)

        self.mask_files1 = []
        for img_path in self.img_files1:
            self.mask_files1.append(os.path.join(Kvasir_folder, 'masks', os.path.basename(img_path)))
        self.mask_files2 = []
        for img_path in self.img_files2:
            self.mask_files2.append(os.path.join(ClinicDB_folder, 'Ground Truth', os.path.basename(img_path)))

    def __getitem__(self, index):

        transform, target_transform, trans = build_transform(self.img_size)

        if index < len(self.img_files1):
            img_path = self.img_files1[index]
            mask_path = self.mask_files1[index]
            data = Image.open(img_path)
            label = Image.open(mask_path).convert('1')
            return transforms.functional.to_tensor(trans(data)), target_transform(
                transforms.functional.to_tensor(trans(label)))
        else:
            index = index - len(self.img_files1)
            img_path = self.img_files2[index]
            mask_path = self.mask_files2[index]
            data = Image.open(img_path)
            label = Image.open(mask_path).convert('1')
            return transforms.functional.to_tensor(trans(data)), target_transform(
                transforms.functional.to_tensor(trans(label)))

    def __len__(self):
        return len(self.img_files1) + len(self.img_files2)


# if __name__ == '__main__':
#     train_ds = KvasirDataSet(
#         "D:/MedicalSeg/Kvasir-SEG/",
#         "D:/MedicalSeg/CVC-ClinicDB/",
#         train_mode=True
#     )
#     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
#     print(len(train_ds))  # 1612
#     print(len(train_loader)) # 51
#
#     valid_ds = KvasirDataSet(
#         "D:/MedicalSeg/Kvasir-SEG/",
#         "D:/MedicalSeg/CVC-ClinicDB/",
#         train_mode=False
#     )
#     valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=32, shuffle=False)
#     print(len(valid_ds))  # 322
#     print(len(valid_loader))  # 11


def build_dataset(args):
    train_ds = KvasirDataSet(
        args.Kvasir_path,
        args.ClinicDB_path,
        train_mode=True
    )

    valid_ds = KvasirDataSet(
        args.Kvasir_path,
        args.ClinicDB_path,
        train_mode=False
    )
    return train_ds, valid_ds