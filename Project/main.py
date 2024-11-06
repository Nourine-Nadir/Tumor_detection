from get_features import extract_features
from train import train
from test import evaluate_model
import json
# HYPERPARAMETERS
with (open('params.json', 'r') as f):
    params = json.load(f)["parameters"]

    PARTS,_nb_features, train_ROOT_FOLDER, train_TARGET_FOLDER,\
     test_ROOT_FOLDER, test_TARGET_FOLDER, TARGET_SIZE, SAVE_FILE, \
    _lr, _fc1_dims, _fc2_dims, _fc3_dims, _epochs, _batch_size\
    =(params[key] for key in
     list(params.keys())
     )


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = extract_features(PARTS,
                                        _nb_features,
                                        train_ROOT_FOLDER,
                                        train_TARGET_FOLDER,
                                        TARGET_SIZE,
                                        SAVE_FILE
                     )

    print(f'-------TRAIN------\nfeatures : {X_train.shape}')
    print(f'labels : {y_train.shape}')

    trained_net, train_loss = train(features=X_train,
          labels=y_train,
          lr=_lr,
          fc1_dims=_fc1_dims,
          fc2_dims=_fc2_dims,
          fc3_dims=_fc3_dims,
          n_epochs=_epochs,
          batch_size=_batch_size)

    print(f'\n-------TEST------\nfeatures : {X_test.shape}')
    print(f'labels : {y_test.shape}')

    accuracy, loss = evaluate_model(trained_net, X_test, y_test, train_loss)
    print(f"\nOverall Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")