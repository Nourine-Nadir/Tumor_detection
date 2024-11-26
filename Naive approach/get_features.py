# IMPORT LIBRARIES
from utils import *
from engine import Engine
from sklearn.model_selection import train_test_split

# LOAD images from folders
def extract_features(h_parts,
                     v_parts,
                     _nb_features,
                     _root_folder,
                     _target_folder,
                     _target_size,
                     _save_file,
                     load_and_resize=True,
                     display=False):

    if load_and_resize:
        print('load and resize')
        images, images_dict = load_images_resize(root_folder= _root_folder,
                                                       output_folder= _target_folder,
                                                       target_size=_target_size,
                                                       save_files=_save_file,
                                                       )
    else:
        images, images_dict = load_images(target_folder= _target_folder,
                                                   )


    engine = Engine(root_folder= _root_folder,
                    output_folder= _target_folder,
                    target_size=_target_size)

    labels =engine.labeling(images, images_dict)
    print(f'labels {labels.shape}')
    splitted_images = engine.split_images(images,v_parts,h_parts)

    nb_imgs = 3
    if display:
        display_images(splitted_images[:nb_imgs*v_parts*h_parts],rows=v_parts*3,cols=h_parts)

    hists = engine.compute_hist(splitted_images, nb_features=_nb_features)
    features = engine.get_features_from_hists(hists,v_parts,h_parts)

    normalized_features = normalize_features(features)

    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels,
        random_state=42,
        test_size=0.2,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test