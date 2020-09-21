import glob
import concurrent.futures
import re
import csv
import os
import traceback

from Features.ScanpathFeatures import scanpath_feature_train, Scanpath_Feature_Names, scanpath_feature_test
from Features.SaliencyFeatures import saliency_feature_train, Saliency_Feature_Names


# from Features import ScanpathFeatures

class Features:
    def __init__(self, scanpaths_dir, images_dir):
        self.__scanpaths_dir = scanpaths_dir
        self.__images_dir = images_dir

    def extract_scanpath_features_train(self, feature_class, dir_to_save):
        extracted_features = []

        scanpaths_file_lst, images_file_lst = sort_scanpath_image_files(self.__scanpaths_dir, self.__images_dir)

        # retrieving content of each text file and extracting features from the scan paths.
        # using a 10 threads to handle each text file.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            extract_feature = {executor.submit(scanpath_feature_train, img_scnpth[0], img_scnpth[1],
                                               feature_class): img_scnpth
                               for img_scnpth in zip(scanpaths_file_lst, images_file_lst)}

            for future in concurrent.futures.as_completed(extract_feature):
                # feature = extract_feature[future]
                try:
                    extracted_features.append(future.result())
                except Exception as exc:
                    print("Error occured: ", exc)

        #  check if csv file exisits
        flchk = os.path.isfile(dir_to_save)

        print("scanpath_featurelen", len(extracted_features))  # debug
        if len(extracted_features) > 0:
            with open(dir_to_save, 'a', newline='') as result_file:
                write_row = csv.writer(result_file, dialect='excel')
                if not flchk:
                    write_row.writerow(Scanpath_Feature_Names)
                for scanpaths in extracted_features:
                    # write_row.writerow(scanpaths)
                    for scanpath in scanpaths:
                        write_row.writerow(scanpath)

    # extracting saliency features for training purposes And Writing to csv file
    def extract_saliency_feature_train(self, feature_class, dir_to_save):
        extracted_features = []

        scanpaths_file_lst, images_file_lst = sort_scanpath_image_files(self.__scanpaths_dir, self.__images_dir)

        # retrieving content of each text file and extracting features from the scan paths.
        # using a 10 threads to handle each text file.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            extract_feature = {executor.submit(saliency_feature_train, img_scnpth[0], img_scnpth[1],
                                               feature_class): img_scnpth
                               for img_scnpth in zip(scanpaths_file_lst, images_file_lst)}

            for future in concurrent.futures.as_completed(extract_feature):
                # feature = extract_feature[future]
                try:
                    extracted_features.append(future.result())
                except Exception as exc:
                    print("Error occured: ", exc)
                    traceback.print_exc()

        #  check if csv file exisits
        flchk = os.path.isfile(dir_to_save)

        if len(extracted_features) > 0:
            with open(dir_to_save, 'a', newline='') as result_file:
                write_row = csv.writer(result_file, dialect='excel')
                if not flchk:
                    write_row.writerow(Saliency_Feature_Names)
                for saliency_feat in extracted_features:
                    # write_row.writerow(scanpaths)
                    for val in saliency_feat:
                        write_row.writerow(val)

    # will extract both the saliency and scanpath features and save it to file
    def extract_all_feautes(self, feature_class, dir_to_save):
        extracted_features = []

        scanpath_file_list, images_file_list = sort_scanpath_image_files(self.__scanpaths_dir, self.__images_dir)
        for i, j in zip(scanpath_file_list, images_file_list):
            saliency_feat = saliency_feature_train(scanpath_fl=i, image_fl=j)
            scanpath_feat = scanpath_feature_train(scanpath_fl=i, image_fl=j, feature_class=feature_class)

            all_feat = list(zip(saliency_feat, scanpath_feat))

            extracted_features.append(all_feat)

            #  check if csv file exisits
            flchk = os.path.isfile(dir_to_save)

            if len(extracted_features) > 0:
                with open(dir_to_save, 'a', newline='') as result_file:
                    write_row = csv.writer(result_file, dialect='excel')
                    if not flchk:
                        write_row.writerow(Scanpath_Feature_Names)
                    for saliency_feat in extracted_features:
                        # write_row.writerow(scanpaths)
                        for val in saliency_feat:
                            write_row.writerow(val)


# calculate the feature values for test data
def extract_features_test(scanpath_fl, image_fl):
    feature_val_list = scanpath_feature_test(scanpath_fl, image_fl)

    return feature_val_list


def sort_scanpath_image_files(scanpaths_dir, images_dir):
    # dir where the list of scanpaths are stored.
    scanpaths_file_lst = glob.glob(scanpaths_dir + "/*.txt")
    scanpaths_file_lst = sorted(scanpaths_file_lst, key=key_func)  # sorting the text files by name

    # dir where the images are stored
    images_file_lst = glob.glob(images_dir + "/*.png")
    images_file_lst = sorted(images_file_lst, key=key_func)  # sorting the images by name

    return scanpaths_file_lst, images_file_lst


def key_func(x):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", x))


if __name__ == "__main__":
    label = "TD"
    feature = Features(scanpaths_dir="C:/Users/Brijesh Prajapati/Documents/Projects/TrainingDataset_YEAR_PROJECT/"
                                     "TrainingData_1/{}".format(label),
                       images_dir="C:/Users/Brijesh Prajapati/Documents/Projects/TrainingDataset_YEAR_PROJECT"
                                  "/TrainingData/Images")
    feature.extract_saliency_feature_train(label, "D:/TrainingDataset_YEAR_PROJECT/TrainingSet_Saliency.csv")
