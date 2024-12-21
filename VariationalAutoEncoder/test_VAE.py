from models import V_AutoEncoder
import torch as T
from utils import *
def test_AE(images,
         latent_dim,
         encoder_model_path,
         full_model_path,
         ):
    model = V_AutoEncoder(input_size=images.shape[-1],latent_dim=latent_dim)
    try:
        model.load_model(full_model_path)
    except:
        print('Model couldn\'t be loaded for testing')
    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(images.astype(np.float32) / 255.0)
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        images = images.cpu().detach().numpy()

        outputs, _, _ = model(images)
        outputs = outputs.cpu().detach().numpy()

    nb_imgs = 6
    indices = np.random.choice(len(images), size=nb_imgs, replace=False)
    display_images(np.squeeze(images)[indices], rows=int(nb_imgs / 2), cols=2, title='Original Images')
    display_images(np.squeeze(outputs)[indices], rows=int(nb_imgs / 2), cols=2, title='Generated Images')
