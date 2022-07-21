from __future__ import print_function, division

import torch
import torch.nn as nn
from efficientnet.model import EfficientNet
from PIL import Image
import numpy as np
from sklearn.utils import compute_class_weight
from torchvision import datasets, models, transforms
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EfficientNet.from_pretrained('efficientnet-b7')
    class_num = 108
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, class_num)
    # model = EfficientNet.from_pretrained('efficientnet-b7').to(device)
    state = torch.load('./pill_ckp.pt')
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    data = transforms.ToTensor()(Image.open("./VAIPE_P_0_0.jpg")).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
    #   val_output = model(data).argmax(dim=1)
        val_output = model(data)
        print(val_output)