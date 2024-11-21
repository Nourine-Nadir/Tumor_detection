from models import AutoEncoder
import numpy as np
import torch as T
import torch.nn as nn
def test(data,
         model_path):
    model = AutoEncoder()
    criterion = nn.MSELoss()

    model.load_model(model_path,map_location=T.device('cuda'))
    print(model)
    images = T.FloatTensor(data.astype(np.float32) / 255.0)

    # Add channel dimension if not present
    if len(images.shape) == 3:
        images = images.unsqueeze(1)  # Add channel dimension (B, 1, H, W)

    outputs = model(images)
    images = images.to(model.device)
    loss = criterion(outputs, images)
    print(f'loss {loss.item()}')
