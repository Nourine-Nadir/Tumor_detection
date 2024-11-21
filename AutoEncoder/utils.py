import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".BMP",".tif")):  # Add or remove file extensions as needed
            img_path = os.path.join(folder_path, filename)
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    images.append(img_array)
            except IOError:
                print(f"Error loading image: {filename}")
    return images
def load_images_from_folders(root_folder,
                             output_folder,
                             save_files= False,
                             target_size=(128, 128)):
    print('Loading from folders ...')
    images = []
    images_dict = {}
    if save_files:
        os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(root_folder):

        label = os.path.basename(root)
        if label and label not in images_dict:  # Only add non-empty labels
            print(f'Procesing {label}')
            images_dict[label] = []
            if save_files:
                label_output_folder = os.path.join(output_folder, label)
                os.makedirs(label_output_folder, exist_ok=True)
        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif','.tif')):
                # Get the full path of the image file
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        img_resized = img.convert('L').resize(target_size)
                        if save_files:
                            output_path = os.path.join(label_output_folder, file)
                            img_resized.save(output_path)

                        img_array = np.array(img_resized)
                        images.append(img_array)
                        images_dict[label].append(img_array)

                except IOError:
                    print(f"Error loading image: {file}")


    return images, images_dict

def load_images_from_folders2(root_folder,

                             ):
    images = []
    images_dict = {}
    print(f'Loading from folders ... {root_folder}')
    for root, dirs, files in os.walk(root_folder):
        label = os.path.basename(root)
        if label and label not in images_dict:  # Only add non-empty labels
            print(f'Procesing {label}')
            images_dict[label] = []

        for file in files:
            # Check if the file is an image by its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif','.tif')):
                # Get the full path of the image file
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:


                        img_array = np.array(img)
                        images.append(img_array)
                        images_dict[label].append(img_array)

                except IOError:
                    print(f"Error loading image: {file}")


    return np.array(images), images_dict


def display_images(images, rows=1, cols=1):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for i, img_array in enumerate(images):
        if i < rows * cols:
            axes[i].imshow(img_array)
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()



def display_hist(images, rows=1, cols=1, nb_features=15):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    axes = axes.flatten()
    for i, img_array in enumerate(images):
        if i < rows * cols:
            # Supposing that the image is in colors
            # img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

            # Supposing that the image is already in grayscale
            img_array = img_array[...,1]
            img = img_array.flatten()


            axes[i].hist(img, bins=nb_features, range=(0, 255))
            axes[i].title.set_text(f'image nb {i}')

    plt.tight_layout()
    plt.show()
    plt.close()


def normalize_features(features):
    normalizer = StandardScaler()
    normalized_features = normalizer.fit_transform(features)
    return normalized_features