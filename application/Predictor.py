from PCHandler import PCHandler
from ImageHandler import ImageHandler
import os
from typing import NoReturn


class Predictor:
    """
    Attributes:
        1)_image_handler = object of type ImageHandler used to take
        care of image processing tasks
        2)_pc_handler = object of type PCHandler used to take care of
        pc-related tasks
        3)_class_threshold = scalar used for classification
        4)_projection_threshold = scalar used for classification
    """
    _image_handler = None
    _pc_handler = None
    _class_threshold = None
    _projection_threshold = None

    @staticmethod
    def __init__(train_path, pc_number, class_threshold=20000, projection_threshold=20000):
        """
        Function that initialises the image handler and the pc handler and sets the classification
        thresholds
        :param train_path: path to training data
        :param pc_number: number of principal components
        :param class_threshold: classification scalar
        :param projection_threshold: classification scalar
        """
        Predictor._image_handler = ImageHandler(train_path)
        Predictor._image_handler.load_image_matrix()
        Predictor._image_handler.load_avg_image()
        Predictor._image_handler.load_c_image_matrix()

        Predictor._pc_handler = PCHandler(pc_number)
        Predictor._pc_handler.load_pc_matrix(Predictor._image_handler.get_c_image_matrix())

        Predictor._class_threshold = class_threshold
        Predictor._projection_threshold = projection_threshold

    @staticmethod
    def make_prediction(image_path) -> str:
        """
        Function that predicts the class of a specified image
        :param image_path: path to the test image
        :return: predicted class label
        """
        test_image = Predictor._image_handler.prepare_image(image_path)
        c_test_image = test_image - Predictor._image_handler.get_avg_image()
        projected_image = Predictor._pc_handler.project_onto_pc_matrix(c_test_image)
        projection_distance = Predictor._pc_handler.compute_proj_distance(c_test_image, projected_image)

        class_scores = Predictor._pc_handler.compute_class_scores(c_test_image,
                                                                  Predictor._image_handler.get_class_average())
        min_score = min(class_scores.values())
        label = [key for key in class_scores if class_scores[key] == min_score]

        if projection_distance < Predictor._projection_threshold and min_score < Predictor._class_threshold:
            return label[0]
        elif min_score > Predictor._class_threshold and projection_distance < Predictor._projection_threshold:
            return "unk"
        else:
            return "object"

    @staticmethod
    def batch_prediction(test_directory, display_statistics=True, verbose_mode=True) -> NoReturn:
        """
        Function used to make predictions in batch mode
        (IMPORTANT All the images have to be in a single folder and they must
        have the following name pattern: -class_name.etc-
        e.g george.1.png, 01.1.png, 01.1.png etc)
        :param test_directory: path to test directory
        :param display_statistics: boolean value used for displaying
        statistics of the algorithm
        :param verbose_mode: boolean value used for printing predicted and
        actual labels
        :return: NoReturn
        """
        correct_count = 0
        batch_size = 0
        for img in os.listdir(test_directory):
            if not img.startswith('.'):
                abs_path = test_directory + '/' + img
                predicted_label = Predictor.make_prediction(abs_path)
                actual_label = img.split('.')[0]
                if verbose_mode:
                    print("Pred: {}, Actual: {}".format(predicted_label, actual_label))
                batch_size += 1
                if predicted_label == actual_label:
                    correct_count += 1
        if display_statistics:
            print("The success rate of the algorithm is {:.2f}%".format((correct_count / batch_size) * 100))




