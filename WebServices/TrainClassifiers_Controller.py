import json
import traceback
from threading import Thread

from Classifiers.MainClassifier import MainClassifier

from flask import Blueprint, request

train_models_bp = Blueprint('train_classifiers', __name__)


@train_models_bp.route('/train_all_models/', methods=['POST'])
def train_all_models():
    try:
        models_params = request.json

        rf_param_dict = {}
        adb_param_dict = {}
        mlp_param_dict = {}
        svm_param_dict = {}

        if len(models_params) == 4:

            for i in models_params[0]:
                rf_param_dict[i['Key']] = i['Value']

            for j in models_params[1]:
                adb_param_dict[j['Key']] = j['Value']

            for k in models_params[2]:
                mlp_param_dict[k['Key']] = k['Value']

            for m in models_params[3]:
                if m['Key'] == "cost_thresh":
                    svm_param_dict[m['Key']] = 1 / (m['Value'])
                else:
                    svm_param_dict[m['Key']] = m['Value']

            print(rf_param_dict)  # debug
            print(adb_param_dict)  # debug
            print(mlp_param_dict)  # debug
            print(svm_param_dict)  # debug

            thread = Thread(target=call_main_classifier, kwargs={'svm_param_dict': svm_param_dict,
                                                                 'rf_param_dict': rf_param_dict,
                                                                 'mlp_param_dict': mlp_param_dict,
                                                                 'adab_param_dict': adb_param_dict})
            thread.start()

            return "Data Received", 200

        else:
            return "Partial Content Received", 206

    except Exception as ex:
        print("error occurred on train_all_models", ex)
        traceback.print_exc()
        return json.dumps({'server error': 'error occurred in Training'}), 500, {'ContentType': 'application/json'}


def call_main_classifier(svm_param_dict, rf_param_dict, mlp_param_dict, adab_param_dict):
    main_classifier = MainClassifier()
    status = main_classifier.train_classifiers(svm_param_dict, rf_param_dict, mlp_param_dict, adab_param_dict)
    return status
