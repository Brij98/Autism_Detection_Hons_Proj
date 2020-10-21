import json
import os
import random
import traceback

import cv2
import pandas as pd
from flask import Flask, request, Blueprint, jsonify, send_file

from Features.AllFeatures import extract_features_test
from Features.SaliencyFeatures import Saliency_Feature_Names_Test
from Features.ScanpathFeatures import Scanpath_Feature_Names_Test
from Features.SaliencyFeatures import compute_saliency_map

from Classifiers.MainClassifier import MainClassifier

classify_samples_bp = Blueprint('classify_samples', __name__)

# data_file_dir = '../ReceivedData'
data_file_dir = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/ReceivedData/"


@classify_samples_bp.route("/classification_service")
def hello():
    return "Hello World"


@classify_samples_bp.route('/classification_request/', methods=['GET', 'POST'])
def classification_request():
    print("upload_data_files Called")  # debug

    print(request.files["file1"])  # debug
    print(request.files["file2"])  # debug

    file_1 = request.files["file1"]
    file_2 = request.files["file2"]

    # filename_1 = os.path.join("D:/TrainingDataset_YEAR_PROJECT/UploadData", file_1.filename)
    # filename_2 = os.path.join("D:/TrainingDataset_YEAR_PROJECT/UploadData", file_2.filename)
    filename_1 = os.path.join(data_file_dir, file_1.filename)
    filename_2 = os.path.join(data_file_dir, file_2.filename)

    # Save the files to dir
    try:
        # try saving the received files
        file_1.save(filename_1)
        file_2.save(filename_2)

        # feature extraction from the scan path file received
        feature_values = extract_features_test(filename_1, filename_2)
        print("feature values", feature_values)  # debug

        classifier = MainClassifier()
        df = pd.DataFrame(feature_values, columns=["fixpoint_count", "total_duration", "mean_duration",
                                                   "total_scanpath_len", "mean_scanpath_len", "mean_dist_centre",
                                                   "mean_dist_mean_coord"])

        classification_dict, prediction = classifier.predict_sample(df)

        print(prediction)

        return classification_dict

    except Exception as ex:
        print(ex)
        traceback.print_exc()
        return json.dumps({'server error': 'error occurred classifying'}), 500, {'ContentType': 'application/json'}

    # return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@classify_samples_bp.route('/classify/', methods=['GET', 'POST'])
def classify():
    filename_1 = ""
    filename_2 = ""
    try:
        print(request.files["file1"])  # debug
        print(request.files["file2"])  # debug

        file_1 = request.files["file1"]
        file_2 = request.files["file2"]

        filename_1 = os.path.join(data_file_dir, file_1.filename)
        filename_2 = os.path.join(data_file_dir, file_2.filename)

        file_1.save(filename_1)
        file_2.save(filename_2)

        # feature extraction from the scan path file received
        feature_values = extract_features_test(filename_1, filename_2)
        print("feature values", feature_values)  # debug

        classifier = MainClassifier()
        df = pd.DataFrame(feature_values,
                          columns=["first_saliency_fixation", "first_above_0.75_max_rank", "first_above_0.9_max_rank",
                                   "saliency_value_mean", "saliency_value_sum", "weighted_duration_sum",
                                   "weighted_duration_mean", "max_saliency_value", "relative_entropy",
                                   "normalized_saliency_scanpath",
                                   "fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                                   "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

        feat_dict = df.to_dict(orient='index')[0]
        print("dict", feat_dict)

        classification_dict, prediction = classifier.predict_sample(df)

        print(prediction)

        list_to_ret = list()
        list_to_ret.append(classification_dict)
        list_to_ret.append(feat_dict)

        try:
            os.remove(filename_1)
            os.remove(filename_2)
        except FileNotFoundError as ex:
            pass

        return jsonify(list_to_ret), 200

    except Exception as ex:
        print(ex)
        traceback.print_exc()
        try:
            os.remove(filename_1)
            os.remove(filename_2)
        except FileNotFoundError as ex:
            pass
        return json.dumps({'server error': 'error occurred classifying'}), 500, {'ContentType': 'application/json'}


@classify_samples_bp.route('/get_saliency_map/', methods=['GET', 'POST'])
def get_saliency_map():
    filename_1 = ""
    filename_3 = ""
    try:

        print(request.files["file1"])  # debug

        file_1 = request.files["file1"]

        filename_1 = os.path.join(data_file_dir, file_1.filename)
        filename_3 = os.path.join(data_file_dir, "salmap.png")

        file_1.save(filename_1)

        img = cv2.imread(filename_1)

        sal_map = compute_saliency_map(img)
        cv2.imwrite(filename_3, sal_map)

        return send_file(filename_3, mimetype='image/png'), 200

    except Exception as ex:
        print(ex)
        traceback.print_exc()
        try:
            os.remove(filename_1)
        except FileNotFoundError as ex:
            pass
        return json.dumps({'server error': 'error occurred classifying'}), 500, {'ContentType': 'application/json'}


@classify_samples_bp.route('/get_feature_values/', methods=['GET', 'POST'])
def get_feature_values():
    filename_1 = ""
    try:
        print(request.files["file1"])  # debug

        file_1 = request.files["file1"]

        filename_1 = os.path.join(data_file_dir, file_1.filename)
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        try:
            os.remove(filename_1)
        except FileNotFoundError as ex:
            pass
        return json.dumps({'server error': 'error occurred classifying'}), 500, {'ContentType': 'application/json'}

# if __name__ == "__main__":
#     app.run(debug=True)
