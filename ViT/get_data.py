from AutoEncoder.train import train
from utils_ViT import *
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, ToTensor

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

def get_data(
                     root_folder,
                     target_folder,
                     target_size,
                     save_file,
                     _display_images,
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
    #
    train_images, test_images, y_train, y_test = train_test_split(
        images, labels,
        random_state=41,
        test_size=0.2,  # 20% for testing
        shuffle=True  # Ensure the data is shuffled
    )

    to_tensor = [Resize((target_size)), ToTensor()]
    transform = Compose(to_tensor)

    train_images_transformed = []
    y_train_transformed = []
    for img, target in zip(train_images, y_train):
        transformed_img, transformed_target = transform(img, target)
        train_images_transformed.append(transformed_img)
        y_train_transformed.append(transformed_target)


    test_images_transformed = []
    y_test_transformed = []
    for img, target in zip(test_images, y_test):
        transformed_img, transformed_target = transform(img, target)

        test_images_transformed.append(transformed_img)
        y_test_transformed.append(transformed_target)
    if _display_images:
        display_images(train_images_transformed[:6],rows=2,cols=3)

    return train_images_transformed, test_images_transformed, y_train_transformed, y_test_transformed