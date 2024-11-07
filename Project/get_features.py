# IMPORT LIBRARIES
from utils import *
from engine import Engine
from sklearn.model_selection import train_test_split

# LOAD images from folders
def extract_features(_parts,
                     _nb_features,
                     _root_folder,
                     _target_folder,
                     _target_size,
                     _save_file,
                     load_and_resize=True,
                     display=False):

    if load_and_resize:
        print('load and resize')
        images, images_dict = load_images_from_folders(root_folder= _root_folder,
                                                       output_folder= _target_folder,
                                                       target_size=_target_size,
                                                       save_files=_save_file)
    else:
        images, images_dict = load_images_from_folders2(root_folder= _target_folder,
                                                   )


    engine = Engine(root_folder= _root_folder,
                    output_folder= _target_folder,
                    target_size=_target_size)

    labels =engine.labeling(images, images_dict)

    splitted_images = engine.split_images(images, _parts)

    nb_imgs = 3
    if display:
        display_images(splitted_images[:nb_imgs*9],rows=len(images[:nb_imgs])*3,cols=_parts)

    hists = engine.compute_hist(splitted_images, nb_features=_nb_features)
    features = engine.get_features_from_hists(hists, parts=_parts)

    normalized_features = normalize_features(features)

    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels,
        test_size=0.2,
         random_state=42,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test