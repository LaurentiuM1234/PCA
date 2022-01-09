import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
from typing import NoReturn
from typing import Union


class ImageHandler:
    """
    Attributes:
        1)_image_matrix = matrix containing every image from all the classes
              in a column vector form
        2)_class_dictionary = dictionary containing the class names as keys and
              image matrices for each class as values
        3)_c_image_matrix = the image matrix centered around the average image
              avg_image = column vector representing the avg. image of the set
        4)_avg_image = column vector representing the average image of the dataset
        5)_image_shape = tuple containing the shape of an image
        6)_dataset_directory = path to directory containing the dataset
    """
    _image_matrix = None
    _c_image_matrix = None
    _avg_image = None
    _image_shape = None
    _class_dictionary = {}
    _dataset_directory = None

    @staticmethod
    def __init__(dataset_directory):
        ImageHandler._dataset_directory = dataset_directory

    # Class getters
    @staticmethod
    def get_image_matrix():
        return ImageHandler._image_matrix

    @staticmethod
    def get_c_image_matrix():
        return ImageHandler._c_image_matrix

    @staticmethod
    def get_avg_image():
        return ImageHandler._avg_image

    @staticmethod
    def get_image_shape():
        return ImageHandler._image_shape

    @staticmethod
    def get_class_dictionary():
        return ImageHandler._class_dictionary

    @staticmethod
    def get_dataset_directory():
        return ImageHandler._dataset_directory

    # Class setters
    @staticmethod
    def set_image_shape(image_shape):
        ImageHandler._image_shape = image_shape

    @staticmethod
    def set_dataset_directory(dataset_directory):
        ImageHandler._dataset_directory = dataset_directory

    # Class loaders
    @staticmethod
    def load_image_matrix() -> NoReturn:
        """
        Function that computes the image matrix, image shape and
        class dictionary for directory path specified at
        initialization and loads them into ImageHandler's attributes
        with the same names.
        :return: NoReturn
        """
        for cls in os.listdir(ImageHandler._dataset_directory):

            if not cls.startswith('.'):
                class_directory = ImageHandler._dataset_directory + '/' + cls
                tmp_image_matrix = None
                for im in os.listdir(class_directory):
                    if not im.startswith('.'):
                        image_path = class_directory + '/' + im
                        image = img.imread(image_path)
                        image = image.astype('float64')

                        if ImageHandler._image_shape is None:
                            ImageHandler.set_image_shape(image.shape)

                        image = image.flatten('C')
                        image = np.asmatrix(image)

                        if tmp_image_matrix is None:
                            tmp_image_matrix = image
                        else:
                            tmp_image_matrix = np.vstack((tmp_image_matrix, image))

                if ImageHandler._image_matrix is None:
                    ImageHandler._image_matrix = tmp_image_matrix
                else:
                    ImageHandler._image_matrix = np.vstack((ImageHandler._image_matrix,
                                                            tmp_image_matrix))

                ImageHandler._class_dictionary[cls] = tmp_image_matrix.T
        ImageHandler._image_matrix = ImageHandler._image_matrix.T

    @staticmethod
    def load_c_image_matrix() -> NoReturn:
        """
        Function that computes the centered image matrix and loads
        it into the ImageHandler's attribute with the same name.
        :return: NoReturn
        """
        if ImageHandler._image_matrix is None:
            print("Unable to load the centered image matrix. "
                  "Please make sure to load the image matrix first!")
        elif ImageHandler._avg_image is None:
            print("Unable to load the centered image matrix. "
                  "Please make sure you load the avg. image first !")
        else:
            ImageHandler._c_image_matrix = ImageHandler._image_matrix - ImageHandler._avg_image

    @staticmethod
    def load_avg_image() -> NoReturn:
        """
        Function that computes the average image and loads it into
        ImageHandler's attribute with the same name.
        :return: NoReturn
        """
        if ImageHandler._image_matrix is None:
            print("Unable to load the avg. image. "
                  "Please make sure to load the image matrix first!")
        else:
            ImageHandler._avg_image = ImageHandler._image_matrix.mean(1)
            avg_img_height = ImageHandler._image_shape[0] * ImageHandler._image_shape[1]
            ImageHandler._avg_image = np.reshape(ImageHandler._avg_image, (avg_img_height, 1))

    # Misc. functions
    @staticmethod
    def plot_image(image) -> NoReturn:
        """
        Function that takes a column vector representing an image and plots it
        :param image: column vector
        :return: NoReturn
        """
        reshaped_image = np.reshape(image, ImageHandler._image_shape)
        plt.imshow(reshaped_image, cmap='gray')
        plt.show()

    @staticmethod
    def image_search(class_name, image_index) -> Union[np.ndarray, None]:
        """
        Function that looks for an image index in the class dictionary
        value specified by the key @class_name
        :param class_name: label/name of the class
        :param image_index: index of the image
        :return: None if the class_name is not in the dictionary or
        the image_index is out of bounds OR column vector representing
        the searched image
        """
        if class_name not in ImageHandler._class_dictionary:
            print("Unknown class name.")
            return None
        else:
            _, image_number = ImageHandler._class_dictionary[class_name].shape
            if image_index > image_number:
                print("The number of images in given class is smaller than the provided"
                      "image index")
                return None
            else:
                return ImageHandler._class_dictionary[class_name][:, image_index - 1]

    @staticmethod
    def get_class_average() -> Union[dict, None]:
        """
        Function that computes the average image for each class
        :return: Dictionary containing each class name and its
        corresponding class average or None if any of the dependencies
        is not satisfied
        """
        class_avg_dictionary = {}
        if ImageHandler._class_dictionary is None:
            print("Class dictionary has not yet been loaded. Unable to compute class average")
            return None
        else:
            for cls in ImageHandler._class_dictionary:
                class_average = (ImageHandler._class_dictionary[cls]).mean(1)
                class_average_height = ImageHandler._image_shape[0] * ImageHandler._image_shape[1]
                class_average = np.reshape(class_average, (class_average_height, 1))
                class_avg_dictionary[cls] = class_average - ImageHandler._avg_image
        return class_avg_dictionary

    @staticmethod
    def prepare_image(image_path) -> np.ndarray:
        """
        Function used to prepare an image for processing
        :param image_path: path to an image
        :return: column vector representing the processed image
        """
        image = plt.imread(image_path)
        im_height, im_width = image.shape
        image = image.astype('float64')
        image = image.flatten('C')
        image = np.asmatrix(image)
        image = np.reshape(image, (im_height * im_width, 1))
        return image
