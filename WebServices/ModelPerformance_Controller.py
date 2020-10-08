import json
import traceback

from flask import Blueprint, jsonify

model_performance_bp = Blueprint('model_performance', __name__)


@model_performance_bp.route('/view_model_performance/', methods=['GET'])
def view_model_performance():
    try:

        svm_report_fl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                        "Classifier_Reports/svm_current_report.json"

        rf_report_fl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                       "Classifier_Reports/rf_current_report.json"

        adb_report_fl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                        "Classifier_Reports/adaboost_current_report.json"

        mlp_report_fl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                        "Classifier_Reports/mlp_current_report.json"

        mc_report_fl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                       "Classifier_Reports/mainmodel_current_report.json"

        # dict_svm_performance = json.loads(svm_report_fl)
        dict_svm_performance = {}
        with open(svm_report_fl, 'r') as read_svm_report:
            dict_svm_performance = json.load(read_svm_report)
        print(dict_svm_performance)  # debug

        # dict_rf_performance = json.loads(rf_report_fl)
        dict_rf_performance = {}
        with open(rf_report_fl, 'r') as read_rf_report:
            dict_rf_performance = json.load(read_rf_report)
        print(dict_rf_performance)  # debug

        # dict_adb_performance = json.loads(adb_report_fl)
        dict_adb_performance = {}
        with open(adb_report_fl, 'r') as read_adb_report:
            dict_adb_performance = json.load(read_adb_report)

        # dict_mlp_performance = json.loads(mlp_report_fl)
        dict_mlp_performance = {}
        with open(mlp_report_fl, 'r') as read_mlp_report:
            dict_mlp_performance = json.load(read_mlp_report)

        # dict_mc_performance = json.loads(mc_report_fl)
        dict_mc_performance = {}
        with open(mc_report_fl, 'r') as read_mc_report:
            dict_mc_performance = json.load(read_mc_report)

        list_to_ret = list()
        list_to_ret.append(dict_svm_performance)
        list_to_ret.append(dict_rf_performance)
        list_to_ret.append(dict_adb_performance)
        list_to_ret.append(dict_mlp_performance)
        list_to_ret.append(dict_mc_performance)

        print(list_to_ret)  # debug

        return jsonify(list_to_ret), 200
    except Exception as ex:
        print("Exception Occured in view_model_performance", ex)
        traceback.print_exc()
        return json.dumps({'server error': 'error occurred in Returning Performance'}), 500, \
               {'ContentType': 'application/json'}
