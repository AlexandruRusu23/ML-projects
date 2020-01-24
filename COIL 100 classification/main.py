from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import multiclass
from sklearn.naive_bayes import GaussianNB
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import time

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_dataset_feature_set(image_paths):
    raw_images = []
    features = []
    labels = []

    for (i, image_path) in enumerate(image_paths):
        image = cv2.imread(image_path)
        label = image_path.split(os.path.sep)[-1].split("_")[0]

        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)

        raw_images.append(pixels)
        features.append(hist)
        labels.append(label)

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(image_paths)))
    
    return (np.array(raw_images), np.array(features), np.array(labels))

if __name__ == '__main__':
    print("[INFO] describing images...")
    image_paths = list(paths.list_images("dataset"))
    
    raw_images, features, labels = extract_dataset_feature_set(image_paths)
    print("[INFO] raw-pixels matrix: {:.2f}MB".format(raw_images.nbytes / (1024 * 1000.0)))
    print("[INFO] hist-features matrix: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))
    
    (train_raw_px_images, test_raw_px_images, train_raw_px_labels, test_raw_px_labels) = train_test_split(raw_images, labels, test_size=0.25)
    (train_hist_features, test_hist_features, train_hist_labels, test_hist_labels) = train_test_split(features, labels, test_size=0.25)
    
    model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    model.fit(train_raw_px_images, train_raw_px_labels)
    acc = model.score(test_raw_px_images, test_raw_px_labels)
    print("[RESULTS] KNN raw pixel accuracy: {:.2f}%".format(acc * 100))

    model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    model.fit(train_hist_features, train_hist_labels)
    acc = model.score(test_hist_features, test_hist_labels)
    print("[RESULTS] KNN histogram accuracy: {:.2f}%".format(acc * 100))

    svm = SVC(kernel='linear', probability=True)
    start_time = time.time()
    clf_raw = multiclass.OneVsRestClassifier(svm).fit(train_raw_px_images, train_raw_px_labels)
    acc = clf_raw.score(test_raw_px_images, test_raw_px_labels)
    print("[RESULTS] SVM raw pixel accuracy {:.2f}%".format(acc*100))
    print("[INFO] Time elapsed for SVM - raw pixels: {:.2f} seconds".format(time.time() - start_time))
    
    start_time = time.time()
    clf_hist = multiclass.OneVsRestClassifier(svm).fit(train_hist_features, train_hist_labels)
    acc = clf_hist.score(test_hist_features, test_hist_labels)
    print("[RESULTS] SVM histogram accuracy {:.2f}%".format(acc*100))
    print("[INFO] Time elapsed for SVM - histogram: {:.2f} seconds".format(time.time() - start_time))

    nb_clf = GaussianNB()
    nb_clf.fit(train_raw_px_images, train_raw_px_labels)
    acc = nb_clf.score(test_raw_px_images, test_raw_px_labels)
    print("[RESULTS] Nayve Bayes raw pixels accuracy: {:.2f}%".format(acc*100))
    
    nb_clf = GaussianNB()
    nb_clf.fit(train_hist_features, train_hist_labels)
    acc = nb_clf.score(test_hist_features, test_hist_labels)
    print("[RESULTS] Nayve Bayes histogram accuracy: {:.2f}%".format(acc*100))
