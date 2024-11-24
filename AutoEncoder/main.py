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

    _latent_dim, ROOT_FOLDER, TARGET_FOLDER,\
    _load_and_resize,_display,  TARGET_SIZE, SAVE_FILE, \
    _lr, _fc1_dims, _fc2_dims, _fc3_dims,_fc4_dims, _epochs, _batch_size, _encoder_model_path,\
    _full_model_path,_load_model, _train, _test \
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

    print(train_images.shape, test_images.shape)
    print(y_train.shape, y_test.shape)

    if _train:
        print('------ Train AutoEncoder ------')
        full_model, trained_encoder, original_images, gen_images = train_AE(data= train_images,
              latent_dim = _latent_dim,
              load_model=_load_model,
              encoder_model_path=_encoder_model_path,
              full_model_path=_full_model_path,
              epochs=_epochs,
              batch_size=_batch_size,
              lr=_lr,)

        nb_imgs = 2
        display_images(np.squeeze(gen_images)[:nb_imgs],rows=int(nb_imgs/2),cols=2)
        display_images(np.squeeze(original_images)[:nb_imgs],rows=int(nb_imgs/2),cols=2)

    if _test:
        print('------ Test AutoEncoder ------')
        test_AE(data= test_images,
             latent_dim=_latent_dim,
             encoder_model_path=_encoder_model_path,
             full_model_path=_full_model_path)


    print('------ Train Naive Net ------')
    trained_net, train_loss = train(features=train_images,
                                    labels=y_train,
                                    lr=_lr,
                                    fc1_dims=_fc1_dims,
                                    fc2_dims=_fc2_dims,
                                    fc3_dims=_fc3_dims,
                                    fc4_dims=_fc4_dims,
                                    latent_dim=_latent_dim,
                                    encoder_model_path=_encoder_model_path,
                                    n_epochs=_epochs*25,
                                    batch_size=_batch_size)

    print('------ Train Naive Net ------')
    accuracy = evaluate_model(net=trained_net,
                              latent_dim=_latent_dim,
                              encoder_model_path=_encoder_model_path,
                              features=test_images,
                              labels=y_test,
                              train_loss=train_loss)