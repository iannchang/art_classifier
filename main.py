import numpy as np
import cv2
from collections import Counter
import os
import cv_utils
import utils
import copy

#Netherlands 1600-1800 (Dutch Golden Age)
#ItalY 1400-1600 (Renaissance)
#UK (1800-1900) (Victorian School)
#France 1800-1900 (Impressionism)



def main():

    eras_to_test = ["dutch", "english", "italian", "french"]
    all_eras = {}
    for era in eras_to_test:
        image_files = os.listdir(era)
        processed = utils.process_era(era, image_files)
        all_eras[era] = processed
    utils.test_independence(all_eras)
    test_data = utils.process_test_images("test_images")
    test_data_1 = utils.process_test_images("test_images")
    all_eras_1 = all_eras.copy()
    test_data_2 = copy.deepcopy(test_data)
    all_eras_2 = all_eras.copy()
    print("Predictions using Naive Bayes MAP")
    utils.predict_using_nb(test_data[0], test_data[1], all_eras, False)
    print("Predictions using Naive Bayes MLE")
    utils.predict_using_nb(test_data_1[0], test_data_1[1], all_eras_1, True)
    acc = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1 , 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    print("Predictions using Softmax Regression")
    utils.predict_using_sr(test_data_2[0], test_data_2[1], all_eras_2, .0001, 8600)

if __name__ == "__main__":
    main()
