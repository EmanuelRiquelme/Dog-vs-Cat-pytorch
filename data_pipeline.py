import os
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class Dog_v_Cat(Dataset):
    def __init__(self, root_dir,transform = None):
      self.root_dir = root_dir
      self.transforms = transforms if transform else transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Resize((28,28)),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
      self.name_files = self.__name_files__()

    def __name_files__(self):
        return [file_name for file_name in os.listdir(f'{self.root_dir}')]

    def __len__(self):
      return len(self.__name_files__())

    def __get_labels__(self,idx):
        category = self.name_files[idx][:3].lower()
        return torch.tensor(1) if category == 'dog' else torch.tensor(0)
    def __getitem__(self, idx):
       file_name = self.name_files[idx]
       img = Image.open(f'{self.root_dir}/{file_name}')
       img = self.transforms(img)
       label =  self.__get_labels__(idx)
       return img,label 
