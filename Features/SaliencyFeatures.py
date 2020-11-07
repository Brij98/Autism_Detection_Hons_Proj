import os

import cv2

import numpy as np

from Features.Feature_Utils import split_scanpaths

Saliency_Feature_Names = ["first_saliency_fixation", "first_above_0.75_max_rank", "first_above_0.9_max_rank",
                          "saliency_value_mean", "saliency_value_sum", "weighted_duration_sum",
                          "weighted_duration_mean", "max_saliency_value", "relative_entropy",
                          "normalized_saliency_scanpath", "feature_class"]
Saliency_Feature_Names_Test = ["first_saliency_fixation", "first_above_0.75_max_rank", "first_above_0.9_max_rank",
                               "saliency_value_mean", "saliency_value_sum", "weighted_duration_sum",
                               "weighted_duration_mean", "max_saliency_value", "relative_entropy",
                               "normalized_saliency_scanpath"]


# for test/experiments purposes only
def sal_test_map(img_path, ASD, TD):
    input_img = cv2.imread(img_path)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    success, saliency_map = saliency.computeSaliency(input_img)

    saliency_map = (saliency_map * 255).astype("uint8")
    saliency_map_1 = cv2.GaussianBlur(saliency_map, (91, 91), 15)

    cv2.imshow("Original Image", input_img)
    cv2.imshow("Saliency Map", saliency_map)
    cv2.imshow("Saliency Map 1", saliency_map_1)
    cv2.imshow("ASD", cv2.imread(ASD))
    cv2.imshow("TD", cv2.imread(TD))
    cv2.waitKey(0)


def compute_saliency_map(input_img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    success, saliency_map = saliency.computeSaliency(input_img)
    saliency_map = (saliency_map * 255).astype("uint8")
    saliency_map = cv2.GaussianBlur(saliency_map, (91, 91), 15)

    return saliency_map


# extracting saliency features for training
def saliency_feature_train(scanpath_fl, image_fl, feature_class=None):
    input_img = cv2.imread(image_fl)

    image_size = input_img.shape[:2]

    scanpath_lst = split_scanpaths(scanpath_fl=scanpath_fl)

    pred_saliency_map = compute_saliency_map(input_img)
    # print("Sal Map", pred_saliency_map.shape) # debug

    feature_val_list = calculate_saliency_features(scanpath_lst, image_size, pred_saliency_map, feature_class)

    return feature_val_list


# extract saliency features for testing
def saliency_feature_test(scanpath_fl, image_fl):
    input_img = cv2.imread(image_fl)

    image_size = input_img.shape[:2]

    scanpath_lst = split_scanpaths(scanpath_fl=scanpath_fl)

    pred_saliency_map = compute_saliency_map(input_img)

    feature_val_list = calculate_saliency_features(scanpath_lst, image_size, pred_saliency_map)

    return feature_val_list


# calculate saliency features
# scanpath_list: list of scanpaths to extract features from
# image_size: size of the image for which the scanpaths were recorded (height, width)
# predicted_saliency_map: predicted saliency map of the image
def calculate_saliency_features(scanpath_list, image_size, predicted_saliency_map, feature_class=None):
    feature_val_list = []
    predicted_saliency_map_cpy = predicted_saliency_map.copy()

    for scanpath in scanpath_list:
        feature_val_name = []
        feature_val = []
        saliency_values = []

        # creating saliency map from the coordinates of the subject's eye data
        actual_saliency_map = np.zeros(image_size)  # creating blank image of row * col

        # print(type(actual_saliency_map))     # debug
        # print(type(scanpath))  # debug

        for coordinate in scanpath:
            # print(coordinate[1])  # debug
            actual_saliency_map[int(coordinate[2]) - 1, int(coordinate[1]) - 1] = 1

            saliency_values.append(predicted_saliency_map_cpy[int(coordinate[2]) - 1,
                                                              int(coordinate[1]) - 1])

        #   actual_saliency_map = cv2.GaussianBlur(actual_saliency_map, (3, 3), 0.5)  # applying gaussian blur
        actual_saliency_map = cv2.GaussianBlur(actual_saliency_map, (109, 109), 18)  # applying gaussian blur

        # value of the first fixation on the predicted saliency map
        feature_val_name.append("first_saliency_fixation")
        feature_val.append(saliency_values[0])

        maximum_val = predicted_saliency_map_cpy.max()  # maximum value in the predicted saliency map
        for maximum_share in [0.75, 0.9]:
            try:
                first_reaching = np.where(saliency_values >= maximum_val * maximum_share)[0][0] + 1
            except IndexError:
                first_reaching = max(20, len(saliency_values) + 1)
            feature_val_name.append("first_above_{}_max_rank".format(maximum_share))
            feature_val.append(first_reaching)

        feature_val_name.append("saliency_value_mean")
        feature_val.append(np.mean(saliency_values))

        feature_val_name.append("saliency_value_sum")
        feature_val.append(np.sum(saliency_values))

        weighted_duration_sum = 0
        for i, j in zip(scanpath['duration'], saliency_values):
            weighted_duration_sum += (i * j)
        feature_val_name.append("weighted_duration_sum")
        feature_val.append(weighted_duration_sum)

        feature_val_name.append("weighted_duration_mean")
        feature_val.append(weighted_duration_sum / len(saliency_values))

        feature_val_name.append("max_saliency_value")
        feature_val.append(np.max(saliency_values))

        # calculating relative entropy
        actual_saliency_map -= actual_saliency_map.min()
        actual_saliency_map /= actual_saliency_map.sum()

        predicted_saliency_map_cpy -= predicted_saliency_map_cpy.min()

        sum_pred_sal_map = predicted_saliency_map_cpy.sum()
        if sum_pred_sal_map == 0:
            predicted_saliency_map_cpy = np.ones_like(predicted_saliency_map_cpy)
            # print("predicted_saliency_map_cpy sum is : ", 0)    # debug

        temp_pred_sal_map = np.true_divide(predicted_saliency_map, sum_pred_sal_map)
        predicted_saliency_map_cpy = temp_pred_sal_map

        epsilon = 7.0 / 3 - 4.0 / 3 - 1

        relative_entropy = (actual_saliency_map * np.log(actual_saliency_map / (predicted_saliency_map_cpy + epsilon) +
                                                         epsilon)).sum()

        feature_val_name.append("relative_entropy")
        feature_val.append(relative_entropy)

        # Calculating the normalized scanpath saliency
        predicted_saliency_map_cpy -= predicted_saliency_map_cpy.mean()
        std_deviation = predicted_saliency_map_cpy.std()
        if std_deviation != 0:
            predicted_saliency_map_cpy /= std_deviation

        feature_val_name.append("normalized_saliency_scanpath")
        feature_val.append(np.mean(predicted_saliency_map_cpy[scanpath['y'].astype('int') - 1,
                                                              scanpath['x'].astype('int') - 1]))

        if feature_class is not None:
            feature_val_name.append("feature_class")
            feature_val.append(feature_class)

        feature_val_list.append(feature_val)

    return feature_val_list


if __name__ == "__main__":
    # pass
    sal_test_map("C:/Users/Brijesh Prajapati/Documents/Projects/TrainingDataset_YEAR_PROJECT/TrainingData"
                 "/Images/7.png", "C:/Users/Brijesh Prajapati/Documents/Projects/TrainingDataset_YEAR_PROJECT"
                                  "/TrainingData/ASD_FixMaps/7_s.png",
                 "C:/Users/Brijesh Prajapati/Documents/Projects/TrainingDataset_YEAR_PROJECT/TrainingData"
                 "/TD_FixMaps/7_s.png")

# References:
#   "Classifying Autism Spectrum Disorder Based on Scanpaths and Saliency", IEEE, Author: Mikhail Startsev,
#   Micheal Dorr"
