import os
import cv2
import pandas as pd
from tqdm import tqdm
import time
import zipfile
import sys

IN_COLAB = 'COLAB_GPU' in os.environ

if IN_COLAB:
    PATH = "/content/input/"
else:
    PATH = "input/"

PATH_DATASET = os.path.join(PATH, "dataset/")
PATH_TRAIN = os.path.join(PATH_DATASET, "train.csv")
PATH_TEST = os.path.join(PATH_DATASET, "test.csv")
TRAIN_PKL = os.path.join(PATH, "train.pkl")
TEST_PKL = os.path.join(PATH, "test.pkl")


def main():

    if not os.path.exists(PATH_DATASET):
        with zipfile.ZipFile("tsp-cv.zip", 'r') as zip_ref:
            zip_ref.extractall(path=PATH)
            zip_ref.close()

    train_df = pd.read_csv(PATH_TRAIN)
    test_df = pd.read_csv(PATH_TEST)

    train_df["image_hist"] = convert_to_histogram(train_df)
    test_df["image_hist"] = convert_to_histogram(test_df)

    train_df.to_pickle(TRAIN_PKL)
    test_df.to_pickle(TEST_PKL)


def convert_to_histogram(source):

    image_hist = []

    for i in tqdm(range(len(source))):
        file_path = os.path.join(PATH_DATASET, source["filename"][i])
        image = cv2.imread(file_path)[:, :, :]
        color = ('b', 'g', 'r')
        histr = []

        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [5, 256])
            histr.append(hist.transpose().squeeze())

        image_hist.append(histr)

    return image_hist


if __name__ == '__main__':
    main()
