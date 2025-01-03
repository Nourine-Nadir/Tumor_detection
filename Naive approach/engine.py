import numpy as np
import pandas as pd
#
class Engine():
    def __init__(self,
                 root_folder:str,
                 output_folder:str,
                 target_size:tuple
                 ):
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.target_size = target_size

    def split_image(self, image, v_parts, h_parts):
        images = []
        # Get height and width of the image
        height, width = image.shape
        # Compute new_height and new_width
        new_height, new_width = height // v_parts, width // h_parts

        # Splitting image to (parts*parts) parts
        # Square splitting
        for i in range(v_parts):
            for j in range(h_parts):
                img = image[i * new_height:(i + 1) * new_height, j * new_width:(j + 1) * new_width]
                images.append(img)
        return images

    def split_images(self, images, v_parts, h_parts):
        splitted_images = []
        for img in images:

            splitted_images.extend(self.split_image(img, v_parts, h_parts))

        return splitted_images

    def compute_hist(self, images, nb_features):
        hists = []
        for i, img_array in enumerate(images):
            img = img_array.flatten()

            hists.extend([np.histogram(img, bins=nb_features)])

        return hists

    def get_features_from_hists(self, hists, v_parts, h_parts):
        return np.stack([
            np.concatenate([hist[0][1:] for hist in hists[i:i + v_parts*h_parts]])
            for i in range(0, len(hists),  v_parts*h_parts)
        ])

    def labeling(self,images, images_dict):
        # LABELING
        labels = np.zeros(len(images))
        idx = 0
        print('Labeling ....')
        for y, key in enumerate(images_dict.keys()):

            print(f'key : {key} : {len(images_dict[key])}')

            labels[idx:idx + len(images_dict[key])] = y
            idx += len(images_dict[key])

        labels = np.array(pd.get_dummies(labels,dtype=np.uint8))
        return labels