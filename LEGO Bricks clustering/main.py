import numpy as np
import matplotlib as plt
import imutils
import cv2
import os
import shutil
import time
from imutils import paths
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

def image_to_feature_vector(image, size=(224, 224)):
    image = np.expand_dims(cv2.resize(image, size), axis=0)    
    image = preprocess_input(image)
    features = np.array(model.predict(image))
    return features

def extract_dataset_feature_set(image_paths):
    img_features = []
    labels = []
    filenames = []

    for (i, image_path) in enumerate(image_paths):
        image = cv2.imread(image_path)
        filepath = image_path
        label = image_path.split(os.path.sep)[-1].split(" ")[0]

        vgg_features = image_to_feature_vector(image)

        img_features.append(vgg_features.flatten())
        labels.append(label)
        filenames.append(filepath)

        if i > 0 and i % 100 == 0:
            print("[INFO] processed {}/{}".format(i, len(image_paths)))
    
    return (np.array(img_features), np.array(labels), np.array(filenames))

def group_images_in_cluster_folders(foldername, labels, filenames):
    try:
        os.makedirs(foldername)
    except OSError:
        pass

    for i, pred in enumerate(labels):
        try:
            os.makedirs('{}{}/'.format(foldername, pred))
        except OSError:
            pass
        shutil.copy(filenames[i], '{}{}/{}'.format(foldername, pred, filenames[i].split(os.path.sep)[-1]))
        if i > 0 and i % 100 == 0:
            print("Processed: {}/{}".format(i, len(labels)))

def main():
    image_paths = list(paths.list_images("dataset"))
    if os.path.exists('np_features_list.txt') and os.path.exists('np_features_list.txt') and \
        os.path.exists('np_features_list.txt'):
        features_list = np.loadtxt('np_features_list.txt')
        labels = np.genfromtxt('np_labels.txt', dtype=str)
        filenames = np.genfromtxt('np_filenames.txt', dtype=str, delimiter='\n')
    else:
        features_list, labels, filenames = extract_dataset_feature_set(image_paths[:1000])
        np.savetxt('np_features_list.txt', features_list)
        np.savetxt('np_labels.txt', labels, fmt='%s')
        np.savetxt('np_filenames.txt', filenames, fmt='%s')

    print("Features extracted")

    n_clusters = len(set(labels))

    print("Run KMeans..")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features_list)
    group_images_in_cluster_folders('output_kmeans/', kmeans.labels_, filenames)
    print("KMeans done..")

    print("Run Agglomerative..")
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    ward.fit(features_list)
    group_images_in_cluster_folders('output_ward/', ward.labels_, filenames)
    print("Agglomerative done..")

if __name__ == '__main__':
    main()
