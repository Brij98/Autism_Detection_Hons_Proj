#   Utils function
import glob
import re
import numpy as np


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


#  splitting scanpaths into list
def split_scanpaths(scanpath_fl):
    scanpaths = np.genfromtxt(scanpath_fl, names=True, case_sensitive='lower', delimiter=',', dtype=np.float)
    scanpath_start = np.where(scanpaths['idx'] == 0)[0]
    scanpath_end = np.append(scanpath_start[1:], len(scanpaths))
    ret_arr = []
    for i, j in zip(scanpath_start, scanpath_end):
        ret_arr.append(scanpaths[i:j])

    return ret_arr
