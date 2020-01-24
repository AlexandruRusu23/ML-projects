import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
import string
import nltk
import sklearn

import seaborn as sns
from pylab import rcParams

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from nltk.corpus import stopwords
from gensim import corpora
from numpy import linalg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('train_full.txt', header=None, sep='\t')
    df.columns=['id',
                'sentence',
                'start_offset',
                'end_offset',
                'target',
                'native_ad',
                'non_native_ad',
                'dif_native_ad',
                'dif_non_native_ad',
                'standard']
    df['sentence'] = df['sentence'].str.lower()
    df['target'] = df['target'].str.lower()

    #remove punctuation
    table = str.maketrans('', '', string.punctuation)
    df['sentence'] = [df['sentence'][row].translate(table) for row in range(len(df['sentence']))]
    df['target'] = [df['target'][row].translate(table) for row in range(len(df['target']))]

    nltk.download('stopwords')
    stop = stopwords.words('english')
    
    df['sentence'] = df['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['target'] = df['target'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    df['sentence'] = [text.split() for text in df['sentence']]
    df['target'] = [text.split() for text in df['target']]
    
    dict_d = corpora.Dictionary(df['sentence'])
    corpus_d = [dict_d.doc2bow(line) for line in df['sentence']]
    corpus_d_vec_norm = [linalg.norm(vec) for vec in corpus_d]
    df['sentence'] = corpus_d_vec_norm

    dict_d = corpora.Dictionary(df['target'])
    corpus_d = [dict_d.doc2bow(line) for line in df['target']]
    corpus_d_vec_norm = [linalg.norm(vec) for vec in corpus_d]
    df['target'] = corpus_d_vec_norm

    df = df.drop(['id', 'dif_native_ad', 'dif_non_native_ad'], axis=1)

    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train, df_valid = train_test_split(df_train, test_size=0.2)

    df_train_0 = df_train.loc[df['standard'] < 0.35]
    df_train_1 = df_train.loc[df['standard'] >= 0.35]
    df_train_0_x = df_train_0.drop(['standard'], axis=1)
    df_train_1_x = df_train_1.drop(['standard'], axis=1)
    df_valid_0 = df_valid.loc[df['standard'] < 0.35]
    df_valid_1 = df_valid.loc[df['standard'] >= 0.35]
    df_valid_0_x = df_valid_0.drop(['standard'], axis=1)
    df_valid_1_x = df_valid_1.drop(['standard'], axis=1)
    df_test_0 = df_test.loc[df['standard'] < 0.35]
    df_test_1 = df_test.loc[df['standard'] >= 0.35]
    df_test_0_x = df_test_0.drop(['standard'], axis=1)
    df_test_1_x = df_test_1.drop(['standard'], axis=1)

    scaler = StandardScaler().fit(df_train_0_x)
    df_train_0_x_rescaled = scaler.transform(df_train_0_x)
    df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
    df_test_0_x_rescaled = scaler.transform(df_test_0_x)
    df_valid_x_rescaled = scaler.transform(df_valid.drop(['standard'], axis = 1))
    df_test_x_rescaled = scaler.transform(df_test.drop(['standard'], axis = 1))

    nb_epoch = 200
    batch_size = 128
    input_dim = df_train_0_x_rescaled.shape[1] #num of predictor variables, 
    encoding_dim = 32
    hidden_dim = int(encoding_dim / 2)
    learning_rate = 1e-3

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(encoding_dim, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')
    cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
                        save_best_only=True,
                        verbose=0)
    tb = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)

    history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                        verbose=1,
                        callbacks=[cp, tb]).history

    valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
   
    test_x_predictions = autoencoder.predict(df_test_x_rescaled)
    mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
    
    threshold_fixed = 0.0005

    pred_y = [1 if e > threshold_fixed else 0 for e in mse]

    output_file = open('auto_results.txt', 'w')

    for iter in pred_y:
        output_file.write(str(iter) + '\n')

    output_file.close()

    #
    # TEST on test.txt in order to submit results
    #

    df_submit = pd.read_csv('test.txt', header=None, sep='\t')
    df_submit.columns=['id',
                'sentence',
                'start_offset',
                'end_offset',
                'target',
                'native_ad',
                'non_native_ad']

    df_submit['sentence'] = df_submit['sentence'].str.lower()
    df_submit['target'] = df_submit['target'].str.lower()

    table = str.maketrans('', '', string.punctuation)
    df_submit['sentence'] = [df_submit['sentence'][row].translate(table) for row in range(len(df_submit['sentence']))]
    df_submit['target'] = [df_submit['target'][row].translate(table) for row in range(len(df_submit['target']))]

    nltk.download('stopwords')
    stop = stopwords.words('english')
    
    df_submit['sentence'] = df_submit['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df_submit['target'] = df_submit['target'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    df_submit['sentence'] = [text.split() for text in df_submit['sentence']]
    df_submit['target'] = [text.split() for text in df_submit['target']]
    
    dict_d = corpora.Dictionary(df_submit['sentence'])
    corpus_d = [dict_d.doc2bow(line) for line in df_submit['sentence']]
    corpus_d_vec_norm = [linalg.norm(vec) for vec in corpus_d]
    df_submit['sentence'] = corpus_d_vec_norm

    dict_d = corpora.Dictionary(df_submit['target'])
    corpus_d = [dict_d.doc2bow(line) for line in df_submit['target']]
    corpus_d_vec_norm = [linalg.norm(vec) for vec in corpus_d]
    df_submit['target'] = corpus_d_vec_norm

    df_ids = df_submit['id'].values
    print(df_ids)
    df_submit = df_submit.drop(['id'], axis=1)
    df_submit_rescaled = scaler.transform(df_submit)

    submit_x_predictions = autoencoder.predict(df_submit_rescaled)

    mse = np.mean(np.power(df_submit_rescaled - submit_x_predictions, 2), axis=1)
    
    threshold_fixed = 0.00005

    pred_submit = [1 if e > threshold_fixed else 0 for e in mse]

    submit_file = open("submit_file.txt", 'w')

    submit_file.write('id,label\n')

    for (i, iter) in enumerate(pred_submit):
        submit_file.write("{},{}\n".format(df_ids[i], iter))
    
    submit_file.close()

if __name__ == '__main__':
    main()
