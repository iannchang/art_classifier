import numpy as np
import cv_utils
from sr_classifier import SoftmaxRegression
from nb_classifier import NaiveBayes
import os
import math

def process_era(era, filename_array):
    processed_data = []
    #colors = define_colors()
    for filename in filename_array:
        filepath = era + "/" + filename
        image = cv_utils.import_image(filepath)
        rgb = cv_utils.find_rgb(image)
        hsv = cv_utils.find_hsv(image)
        rgb = np.asarray(rgb)
        hsv = np.asarray(hsv)
        #color = find_most_prominent(colors, image)
        processed_data.append(np.hstack((rgb,hsv)))
    return np.asarray(processed_data)

def test_independence(all_eras):
# independence for continuous random variables
#ex: calculate the E[r | dutch] and see if it is different from E[r].
#If they are equal then we can see that they are independent
#note: ranges are [0,255] for rgb. [0,179] for h, [0,255] for s, and [0,255] for v
    era_averages = {}
    overall_totals = np.zeros(6)
    overall_painting_counter = 0
    for era in all_eras:
        totals = np.zeros(6)
        painting_counter = 0
        for painting in all_eras[era]:
            i = 0
            for value in painting:
                totals[i] += value
                overall_totals[i]+=value
                i+=1
            painting_counter+=1
        overall_painting_counter+=painting_counter
        averages = totals/painting_counter
        era_averages[era] = averages
    overall_average = overall_totals/overall_painting_counter
    var_names = ["r", "g", "b", "h", "s", "v"]
    for era in era_averages:
        counter = 0
        for expectation in era_averages[era]:
            print("E["+var_names[counter]+"|"+era + "] = " + str(expectation))
            counter+=1
    counter = 0
    for average in overall_average:
        print("E[" + var_names[counter] + "] = " + str(average))
        counter+=1

def process_test_images(filepath):
    image_eras = os.listdir(filepath)
    test_features = []
    test_labels = []
    for era in image_eras:
        image_files = os.listdir(filepath+"/"+era)
        for filename in image_files:
            full_path = filepath + "/" + era + "/" + filename
            image = cv_utils.import_image(full_path)
            rgb = np.asarray(cv_utils.find_rgb(image))
            hsv = np.asarray(cv_utils.find_hsv(image))
            test_features.append(np.hstack((rgb,hsv)))
            test_labels.append(era)
    return (test_labels, np.asarray(test_features))

def process_training_data(all_eras):
    train_features = []
    train_labels = []
    discrete_labels_dict = {"dutch":0, "english":1, "french":2, "italian":3}
    for era in all_eras:
        for painting in all_eras[era]:
            train_features.append(painting)
            train_labels.append(discrete_labels_dict[era])
    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)
    return (train_features, train_labels,)

def bucket_data(test_features, training_features):
    #bucket continuous features variables to make them discrete.
    row_counter = 0
    for features in test_features:
        column_counter = 0
        for feature in features:
            test_features[row_counter][column_counter] = math.floor(feature/17)
            column_counter+=1
        row_counter+=1
    row_counter = 0
    for features in training_features:
        column_counter = 0
        for feature in features:
            training_features[row_counter][column_counter] = math.floor(feature/17)
            column_counter+=1
        row_counter+=1
    return (test_features, training_features)


def predict_using_nb(test_labels, test_features, all_eras, use_mle):
    nbclf = NaiveBayes(use_mle)
    training_data = process_training_data(all_eras)
    bucketed_data = bucket_data(test_features, training_data[0])
    test_features = bucketed_data[0]
    train_features = bucketed_data[1]
    train_labels = training_data[1]
    nbclf.fit(train_features, train_labels)
    preds = nbclf.predict(test_features)
    name_dict = {"dutch":0,"english":1,"french":2,"italian":3}
    correct_predictions = 0
    test_labels_in_num_form = []
    for i in range(len(preds)):
        test_labels_in_num_form.append(name_dict[test_labels[i]])
        if test_labels_in_num_form[i] == preds[i]:
            correct_predictions+=1
    test_labels_in_num_form = np.asarray(test_labels_in_num_form)
    print("Test Labels: " + str(test_labels_in_num_form))
    print("Predictions: " + str(preds))
    print("# of Correct Predictions: "+ str(correct_predictions))
    print("# of Incorrect Predictions: "+str((len(preds)-correct_predictions)))
    print("% Correct: " + str((correct_predictions/(len(preds)))))

def predict_using_sr(test_labels, test_features, all_eras, learning_rate, steps):
    srclf = SoftmaxRegression(learning_rate,steps)
    training_data = process_training_data(all_eras)
    srclf.fit(training_data[0], training_data[1], 4)
    preds = srclf.predict(test_features)
    name_dict = {"dutch":0,"english":1,"french":2,"italian":3}
    correct_predictions = 0
    test_labels_in_num_form = []
    for i in range(len(preds)):
        test_labels_in_num_form.append(name_dict[test_labels[i]])
        if test_labels_in_num_form[i] == preds[i]:
            correct_predictions+=1
    test_labels_in_num_form = np.asarray(test_labels_in_num_form)
    print("Test Labels: " + str(test_labels_in_num_form))
    print("Predictions: " + str(preds))
    print("# of Correct Predictions: "+ str(correct_predictions))
    print("# of Incorrect Predictions: "+str((len(preds)-correct_predictions)))
    print("% Correct: " + str((correct_predictions/(len(preds)))))
