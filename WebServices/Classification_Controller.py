import json
import os
import traceback

import pandas as pd
from flask import Flask, request

from Features.AllFeatures import extract_features_test

from Classifiers.MainClassifier import MainClassifier

app = Flask(__name__)

data_file_dir = '../ReceivedData'


@app.route("/classification_service")
def hello():
    return "Hello World"


@app.route('/classification_service/classification_request', methods=['GET', 'POST'])
def classification_request():
    print("upload_data_files Called")  # debug

    print(request.files["file1"])  # debug
    print(request.files["file2"])  # debug

    file_1 = request.files["file1"]
    file_2 = request.files["file2"]

    filename_1 = os.path.join("D:/TrainingDataset_YEAR_PROJECT/UploadData", file_1.filename)
    filename_2 = os.path.join("D:/TrainingDataset_YEAR_PROJECT/UploadData", file_2.filename)

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


if __name__ == "__main__":
    app.run(debug=True)
