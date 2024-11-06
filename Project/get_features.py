# IMPORT LIBRARIES
from utils import *
from engine import Engine


# LOAD images from folders
def extract_features(_parts,
                     _nb_features,
                     _root_folder,
                     _target_folder,
                     _target_size,
                     _save_file,
                     display=False):


    # images, images_dict = load_images_from_folders(root_folder= _root_folder,
    #                                                output_folder= _target_folder,
    #                                                target_size=_target_size,
    #                                                save_files=_save_file)

    images, images_dict = load_images_from_folders2(root_folder= _target_folder,
                                                   )


    engine = Engine(root_folder= _root_folder,
                    output_folder= _target_folder,
                    target_size=_target_size)

    labels =engine.labeling(images, images_dict)

    splitted_images = engine.split_images(images, _parts)

    nb_imgs = 3
    if display:
        display_images(splitted_images[:nb_imgs*4],rows=len(images[:nb_imgs])*2,cols=2)

    hists = engine.compute_hist(splitted_images, nb_features=_nb_features)
    features = engine.get_features_from_hists(hists, parts=_parts)

    normalized_features = normalize_features(features)
    return normalized_features, labels