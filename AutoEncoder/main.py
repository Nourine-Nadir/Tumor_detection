import numpy as np
from sklearn.model_selection import train_test_split
from test import evaluate_model
from get_data import get_data
from train_AE import train_AE
from test_AE import test_AE
from train import train
from utils import *
import json
# HYPERPARAMETERS
with (open('params.json', 'r') as f):
    params = json.load(f)["parameters"]

    ROOT_FOLDER, TARGET_FOLDER,\
    _load_and_resize,_display,  TARGET_SIZE, SAVE_FILE, \
    _lr, _fc1_dims, _fc2_dims, _fc3_dims,_fc4_dims, _latent_dim, _epochs, _batch_size, _encoder_model_path,\
    _full_model_path, _naive_model_path, _load_model,_train_AE, _test_AE, _train, _test \
    =(params[key] for key in
     list(params.keys())
     )


if __name__ == '__main__':

    train_images, test_images, y_train, y_test =\
                     get_data(root_folder=ROOT_FOLDER,
                     target_folder=TARGET_FOLDER,
                     target_size=TARGET_SIZE,
                     save_file=SAVE_FILE,
                     load_and_resize=_load_and_resize)

    config = {
        "lr": _lr,
        "fc1_dims": _fc1_dims,
        "fc2_dims": _fc2_dims,
        "fc3_dims": _fc3_dims,
        "fc4_dims": _fc4_dims,
        "latent_dim": _latent_dim,
        "epochs": _epochs,
        "batch_size": _batch_size,
    }

    if _train_AE:
        print('------ Train AutoEncoder ------')
        full_model, trained_encoder, original_images, gen_images = train_AE(data= train_images,
              latent_dim = _latent_dim,
              load_model=_load_model,
              encoder_model_path=_encoder_model_path,
              full_model_path=_full_model_path,
              epochs=_epochs,
              batch_size=_batch_size,
              lr=_lr,)

        nb_imgs = 5
        display_images(np.squeeze(gen_images)[:nb_imgs],rows=int(nb_imgs/2),cols=2)
        display_images(np.squeeze(original_images)[:nb_imgs],rows=int(nb_imgs/2),cols=2)

    if _test_AE:
        print('------ Test AutoEncoder ------')
        test_AE(data= test_images,
             latent_dim=_latent_dim,
             encoder_model_path=_encoder_model_path,
             full_model_path=_full_model_path)

    train_loss = None
    if _train:
        print('------ Train Naive Net ------')
        trained_net, train_loss = train(features=train_images,
                                        labels=y_train,
                                        config=config,
                                        encoder_model_path=_encoder_model_path,
                                        net_model_path=_naive_model_path,

                                        )

    if _test :
        print('------ Test Naive Net ------')
        accuracy = evaluate_model(net_model_path=_naive_model_path,
                                  config=config,
                                  latent_dim=_latent_dim,
                                  encoder_model_path=_encoder_model_path,
                                  features=test_images,
                                  labels=y_test,
                                  train_loss=train_loss)