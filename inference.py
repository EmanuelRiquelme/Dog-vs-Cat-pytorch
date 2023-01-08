import torch
from torchvision import transforms
import os
from model import LeNet
from PIL import Image
transform_img =  transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((28,28)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
labels = ['cat','dog']

cifar_model = LeNet(2)
def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/dog_cat.pt'))

def prediction(img_name,model,transform = transform_img,labels = labels,device = 'cpu'):
    img = Image.open(f'test/{img_name}.jpg')
    img = transform(img).to(device).unsqueeze(0)
    pred = torch.argmax(model(img),-1)
    return labels[pred]
