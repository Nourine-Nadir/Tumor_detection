from get_data import get_data
import json
import numpy as np
from train import train
from test import test


# HYPERPARAMETERS
with (open('params.json', 'r') as f):
    params = json.load(f)["parameters"]

    ROOT_FOLDER, TARGET_FOLDER,\
    _load_and_resize,  TARGET_SIZE, SAVE_FILE, _display_images, \
     _emb_dim, _mlp_dim, _dropout, _patch_size, _n_heads, _n_layers, \
     _lr, _epochs, _batch_size,_model_path, _load_model, \
    _train, _test \
    =(params[key] for key in
     list(params.keys())
     )


if __name__ == '__main__':

    train_images, test_images, y_train, y_test = \
                     get_data(root_folder=ROOT_FOLDER,
                     target_folder=TARGET_FOLDER,
                     target_size=TARGET_SIZE,
                     save_file=SAVE_FILE,
                     _display_images=_display_images,
                     load_and_resize=_load_and_resize)

    print('images : ------- ',np.array(train_images).shape)
    config = {
        "emb_dim": _emb_dim,
        "mlp_dim": _mlp_dim,
        "dropout": _dropout,
        "patch_size": _patch_size,
        "n_heads": _n_heads,
        "n_layers": _n_layers,
        "out_dim": np.unique(y_train),
        "lr": _lr,
        "epochs": _epochs,
        "batch_size": _batch_size,
    }

    if _train:
        model = train(
            images=train_images,
            labels=y_train,
            config=config,
            model_path=_model_path,
            load_model=_load_model,
        )

    if _test:
        test(
            model=model,
            images=test_images,
            labels=y_test,
            config=config,
            model_path=_model_path,
            load_model=_load_model,
        )
