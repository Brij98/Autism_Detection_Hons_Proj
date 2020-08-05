import glob
import concurrent.futures
import re
import csv
import os
from Features.ScanpathFeatures import extract_scanpath_feature, Scanpath_Feature_Names


class Features:
    def __init__(self, scanpaths_dir, images_dir):
        self.scanpaths_dir = scanpaths_dir
        self.images_dir = images_dir

    def extract_scanpath_features_train(self, feature_class, dir_to_save):
        extracted_features = []

        scanpaths_lst = glob.glob(self.scanpaths_dir + "/*.txt")
        scanpaths_lst = sorted(scanpaths_lst, key=key_func)

        images_lst = glob.glob(self.images_dir + "/*.png")
        images_lst = sorted(images_lst, key=key_func)

        # retrieving content of each text file and extracting features from the scan paths.
        # using a 10 threads to handle each text file.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            extract_feature = {executor.submit(extract_scanpath_feature, img_scnpth[0], img_scnpth[1],
                                               feature_class): img_scnpth
                               for img_scnpth in zip(scanpaths_lst, images_lst)}

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

    def extract_scanpath_features_test(self):
        pass


def key_func(x):
    nondigits = re.compile("\D")
    return int(nondigits.sub("", x))


if __name__ == "__main__":
    feature = Features(scanpaths_dir="D:/TrainingDataset_YEAR_PROJECT/TrainingData/TD", images_dir=
    "D:/TrainingDataset_YEAR_PROJECT/TrainingData/Images")
    feature.extract_scanpath_features_train("TD", "D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
