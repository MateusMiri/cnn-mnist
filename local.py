import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import copy
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((28,28), antialias=True),
    torchvision.transforms.Grayscale(num_output_channels=1)
])

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    lenet = torch.load('modelo.pth')
else:
    device = torch.device("cpu")
    lenet = torch.load('modelo.pth', map_location=torch.device('cpu'))
    print("No Cuda Available")


def inference_local_image(path, model, device):
    img = Image.open(path).convert(mode="L")
    img = img.resize((28, 28))
    x = (255 - np.expand_dims(np.array(img), -1))/255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy(), x
    
def inference_web_image(path, model, device):
    r = requests.get(path)
    with BytesIO(r.content) as f:
        img = Image.open(f).convert(mode="L")
        img = img.resize((28, 28))
        x = (255 - np.expand_dims(np.array(img), -1))/255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy(), x
    

# Inferir com Imagem da Web
path = "./Test/8/test.png"
pred, x = inference_local_image(path, lenet, device=device)
plt.imshow(x.squeeze(-1), cmap="gray")
pred_idx = np.argmax(pred)
print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx]*100} %")
# -------------------------


# Inferir com Imagem Local
path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq2h_xaDtemhitxk1AhEyzc5mYQu17d3Qb9Q&s"
pred, x = inference_web_image(path, lenet, device=device)
plt.imshow(x.squeeze(-1), cmap="gray")
pred_idx = np.argmax(pred)
print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx]*100} %")
# -------------------------