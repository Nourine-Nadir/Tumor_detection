from get_data import get_data
from train import train
from test import evaluate_model
import json
from sklearn.model_selection import train_test_split
# HYPERPARAMETERS
with (open('params.json', 'r') as f):
    params = json.load(f)["parameters"]

    ROOT_FOLDER, TARGET_FOLDER,\
    _load_and_resize,  TARGET_SIZE, SAVE_FILE, \
    _lr, _fc1_dims, _fc2_dims, _fc3_dims,_fc4_dims, _epochs, \
    _batch_size,_model_path,  _load_model, _train, _test \
    =(params[key] for key in
     list(params.keys())
     )


if __name__ == '__main__':

    train_images, test_images, y_train, y_test = \
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
        "epochs": _epochs,
        "batch_size": _batch_size,
    }


    if _train:
        print('------ Train Naive Net ------')
        embedded_features, results = train(features=train_images,
                                        labels=y_train,
                                        config=config,
                                        model_path=_model_path,
                                        load_model=_load_model
                                        )
    if _test:
        print('------ Test Naive Net ------')
        accuracy = evaluate_model(model_path=_model_path,
                                                    features=test_images ,
                                                    labels=y_test,
                                                    batch_size=_batch_size,
                                        )

