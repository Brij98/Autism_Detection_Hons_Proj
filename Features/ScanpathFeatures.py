import numpy as np
import cv2
import math
from Features.Feature_Utils import split_scanpaths

Scanpath_Feature_Names = ["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                          "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord", "feature_class"]

Scanpath_Feature_Names_Test = ["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                          "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"]

# Scanpath_Feature_Names = ["fixpoint_count", "mean_duration", "total_scanpath_len",
#                           "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord", "feature_class"]
#
# Scanpath_Feature_Names_Test = ["fixpoint_count", "mean_duration", "total_scanpath_len",
#                           "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"]


# extracting scanpath feature from the text file
def scanpath_feature_train(scanpath_fl, image_fl, feature_class=None):
    image_size = cv2.imread(image_fl).shape
    scanpath_lst = split_scanpaths(scanpath_fl=scanpath_fl)

    feature_val_list = calculate_scan_path_features(scanpath_lst, image_size, feature_class)

    return feature_val_list


# extract scan path features for test data
def scanpath_feature_test(scanpath_fl, image_fl):
    image_size = cv2.imread(image_fl).shape
    scanpath_lst = split_scanpaths(scanpath_fl=scanpath_fl)

    feature_val_list = calculate_scan_path_features(scanpath_lst, image_size)

    return feature_val_list


# calculate the features from the scan path received.
def calculate_scan_path_features(scan_path_list, image_size, feature_class=None):
    feature_val_list = []

    for scanpath in scan_path_list:
        feature_name = []
        feature_val = []

        #  fixation count
        feature_name.append("fixpoint_count")
        feature_val.append(len(scanpath))

        #  total duration
        feature_name.append("total_duration")
        feature_val.append(np.sum(scanpath['duration']))

        #  average duration
        feature_name.append("mean_duration")
        feature_val.append(np.mean(scanpath['duration']))

        #  calculating the euclidean distance
        # x_coords = scanpath['x']
        # x_coords = np.diff(x_coords)
        # y_coords = scanpath['y']
        # y_coords = np.diff(y_coords)

        x_coords = np.zeros(len(scanpath['x']))
        for i in range(len(scanpath['x']) - 1):
            x_coords[i] = scanpath['x'][i + 1] - scanpath['x'][i]

        y_coords = np.zeros(len(scanpath['y']))
        for i in range(len(scanpath['y']) - 1):
            y_coords[i] = scanpath['y'][i + 1] - scanpath['y'][i]

        amplitudes = []
        for x, y in zip(x_coords, y_coords):
            amplitudes.append(math.sqrt(x ** 2) + (y ** 2))
        np.round(amplitudes, 9)

        #  total length of scanpath (sum  of amplitudes)
        feature_name.append("total_scanpath_len")
        feature_val.append(np.sum(amplitudes))

        #  average length of scanpath
        feature_name.append("mean_scanpath_len")
        if len(amplitudes) > 0:
            feature_val.append(np.mean(amplitudes))
        else:
            feature_val.append(0.0)

        #  calculating distance from the centre of the image
        #  AND
        #  calculating distance to the average scanpath coordinate
        x_coords = scanpath['x']
        y_coords = scanpath['y']

        x_coord_avg = np.mean(scanpath['x'])
        y_coord_avg = np.mean(scanpath['y'])

        dist_to_centre = []
        avg_dist_coord = []
        for x, y in zip(x_coords, y_coords):
            dist_to_centre.append(math.sqrt((x - image_size[1] / 2) ** 2 + (y - image_size[0] / 2) ** 2))
            avg_dist_coord.append(math.sqrt((x - x_coord_avg) ** 2 + (y - y_coord_avg) ** 2))
        np.round(dist_to_centre, 9)
        np.round(avg_dist_coord, 9)

        feature_name.append("mean_dist_centre")
        feature_val.append(np.mean(dist_to_centre))

        feature_name.append("mean_dist_mean_coord")
        feature_val.append(np.mean(avg_dist_coord))

        if feature_class is not None:
            feature_name.append("feature_class")
            feature_val.append(feature_class)

        feature_val_list.append(feature_val)

    return feature_val_list

# References:
#   "Classifying Autism Spectrum Disorder Based on Scanpaths and Saliency", IEEE, Author: Mikhail Startsev, Micheal Dorr
