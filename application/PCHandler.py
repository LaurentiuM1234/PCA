import numpy as np
from typing import NoReturn
from typing import Union


class PCHandler:
    """
    Attributes:
        1)_pc_matrix = the matrix containing the principal components
        of the centered image matrix
        2)_pc_number = number of principal components to be taken into
        account
    """
    _pc_matrix = None
    _pc_number = None

    @staticmethod
    def __init__(pc_number):
        PCHandler._pc_number = pc_number

    @staticmethod
    def get_pc_number():
        return PCHandler._pc_number

    @staticmethod
    def get_pc_matrix():
        return PCHandler._pc_matrix

    @staticmethod
    def get_normed_pc_matrix() -> Union[np.ndarray, None]:
        """
        Function used to compute the normalised version of the
        principal component matrix
        :return: None if any of the dependencies has not been met or
        the normalised pc matrix
        """
        if PCHandler._pc_matrix is None:
            print("PC matrix has not yet been loaded. Fail")
            return None
        else:
            normed_pc_matrix = PCHandler._pc_matrix
            for i in range(0, PCHandler._pc_number):
                normed_pc_matrix[:, i] = PCHandler._pc_matrix[:, i] / np.linalg.norm(PCHandler._pc_matrix[:, i])

            return normed_pc_matrix

    @staticmethod
    def load_pc_matrix(c_image_matrix) -> NoReturn:
        """
        Function that computes the pc matrix and loads it into
        PCHandler's attribute with the same name
        :param c_image_matrix: centered image matrix
        :return: NoReturn
        """
        S = np.matmul(c_image_matrix.T, c_image_matrix)
        v_matrix, sig, _ = np.linalg.svd(S)
        PCHandler._pc_matrix = np.matmul(c_image_matrix, v_matrix)
        PCHandler._pc_matrix = PCHandler._pc_matrix[:, 0:PCHandler._pc_number]

    @staticmethod
    def project_onto_pc_matrix(image) -> np.ndarray:
        """
        Function used to project a centered image onto the
        face space
        :param image: column vector representing an image
        :return: projected image
        """
        normed_pc_matrix = PCHandler.get_normed_pc_matrix()
        weight_matrix = np.matmul(normed_pc_matrix.T, image)
        return np.matmul(normed_pc_matrix, weight_matrix)

    @staticmethod
    def compute_proj_distance(image, projected_image) -> float:
        """
        Function used to compute the distance from the centered image
        to its projection onto the face space
        :param image: column vector representing an image
        :param projected_image: column vector representing the projected image
        :return: distance between the 2 column vectors
        """
        return np.linalg.norm(image - projected_image)

    @staticmethod
    def compute_class_scores(image, class_average) -> dict:
        """
        Function that computes the distance between the "projected"
        centered image and the "projected" average image for each of the
        classes
        :param image: column vector representing an image
        :param class_average: dictionary containing the class labels and
        class averages
        :return: dictionary containing the class labels and distances between
        the 2 "projections"
        """
        class_scores = {}
        normed_pc_matrix = PCHandler.get_normed_pc_matrix()
        for cls in class_average:
            class_scores[cls] = np.linalg.norm(np.matmul(normed_pc_matrix.T, image - class_average[cls]))
        return class_scores
