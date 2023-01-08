import torch
import torch.optim as optim
import torch.nn as nn
from model import LeNet
from data_pipeline import Dog_v_Cat
from torch.utils.data import DataLoader
from tqdm import trange
import os 
data = Dog_v_Cat('train')
data = DataLoader(data,batch_size = 512,drop_last = True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

loss_fn = nn.CrossEntropyLoss()
model = LeNet(num_clases= 2).to(device)
lr = .005
opt = optim.Adam(model.parameters(),lr = lr)
epochs = 70
def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/dog_cat.pt'))

def save_model(model):
    torch.save(model.state_dict(),'dog_cat.pt')

def train_pipeline(data_loader = data,loss_fn = loss_fn,opt = opt,
                    model = model,device = device,epochs = epochs):
    for epoch in  (t := trange(epochs)):
        it = iter(data_loader)
        for _ in range(len(data_loader)):
            opt.zero_grad()
            img,label = next(it)
            img,label = img.to(device),label.to(device) 
            pred = model(img) 
            loss = loss_fn(pred,label)
            loss.backward()
            opt.step()
        t.set_description(f'loss_gen: {loss.item():.2f}')

if __name__ == '__main__':
    #load_model(model)
    train_pipeline()
    save_model(model)
