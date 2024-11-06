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
    features, labels = extract_features(PARTS,
                                        _nb_features,
                                        train_ROOT_FOLDER,
                                        train_TARGET_FOLDER,
                                        TARGET_SIZE,
                                        SAVE_FILE
                     )

    print(f'features : {features.shape}')
    trained_net = train(features=features,
          labels=labels,
          lr=_lr,
          fc1_dims=_fc1_dims,
          fc2_dims=_fc2_dims,
          fc3_dims=_fc3_dims,
          n_epochs=_epochs,
          batch_size=_batch_size)

    features, labels = extract_features(PARTS,
                                        _nb_features,
                                        test_ROOT_FOLDER,
                                        test_TARGET_FOLDER,
                                        TARGET_SIZE,
                                        SAVE_FILE
                                        )
    accuracy, loss = evaluate_model(trained_net, features, labels)
    print(f"\nOverall Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")