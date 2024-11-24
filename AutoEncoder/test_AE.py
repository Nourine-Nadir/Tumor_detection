from AutoEncoder.models import Encoder
from models import AutoEncoder
import numpy as np
import torch as T
import torch.nn as nn
def test_AE(data,
         latent_dim,
         encoder_model_path,
         full_model_path,
         ):
    encoder = Encoder(latent_dim=latent_dim)

    encoder.load_model(encoder_model_path,map_location=T.device('cuda'))

    images = T.FloatTensor(data.astype(np.float32) / 255.0)

    # Add channel dimension if not present
    if len(images.shape) == 3:
        images = images.unsqueeze(1)  # Add channel dimension (B, 1, H, W)

    images = images.to(encoder.device)
    print(f'images shape in testing : {images.shape}')
    outputs = encoder(images)
    print(f'latent dim shape : {outputs.shape}')
