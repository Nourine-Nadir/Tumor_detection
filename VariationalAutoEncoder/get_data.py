from utils import *
from sklearn.model_selection import train_test_split
def get_data(
                     root_folder,
                     target_folder,
                     target_size,
                     save_file,
                     load_and_resize=True,
                     ):
    if load_and_resize:
        print('load and resize')
        images, images_dict = load_images_resize(root_folder=root_folder,
                                                 output_folder=target_folder,
                                                 target_size=target_size,
                                                 save_files=save_file,)
    else:
        images, images_dict = load_images(target_folder=target_folder,
                                          )

    labels = labeling(images, images_dict)

    train_images, test_images, y_train, y_test = train_test_split(
        images, labels,
        random_state=41,
        test_size=0.2,  # 20% for testing
        shuffle=True  # Ensure the data is shuffled
    )

    return train_images, test_images, y_train, y_test