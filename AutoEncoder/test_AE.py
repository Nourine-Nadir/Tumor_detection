from models import AutoEncoder
import torch as T
from utils import *
def test_AE(data,
         latent_dim,
         encoder_model_path,
         full_model_path,
         ):
    model = AutoEncoder(latent_dim=latent_dim)

    model.load_model(full_model_path)

    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(data.astype(np.float32) / 255.0)
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        images = images.cpu().detach().numpy()

        outputs = model(images).cpu().detach().numpy()

    nb_imgs = 5
    indices = np.random.choice(len(images), size=nb_imgs, replace=False)
    print(indices)
    display_images(np.squeeze(images)[indices], rows=int(nb_imgs / 2), cols=2)
    display_images(np.squeeze(outputs)[indices], rows=int(nb_imgs / 2), cols=2)
