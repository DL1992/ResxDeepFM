from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.feature_selection import *
import pickle


def load_preprocess():
    label_encoders = []
    with open('lbl_encoders.pkl', "rb") as f:
        for _ in range(pickle.load(f)):
            label_encoders.append(pickle.load(f))
    with open('mms.pkl', 'rb') as g:
        mms = pickle.load(g)
    return mms, label_encoders


def preprocess_data(input_path, output_path, sparse_features, dense_features, mms, sparse_missing_val):
    count = 0
    for df in pd.read_csv(input_path, sep='\t', chunksize=100000, header=None, names=['label'] + cols):
        count += len(df)
        print(f'\rpreprocess {count}', end='')
        df[sparse_features] = df[sparse_features].fillna(sparse_missing_val)
        df[dense_features] = df[dense_features].fillna(0)

        for i, feat in enumerate(sparse_features):
            lbe = label_encoders[i]
            df[feat] = df[feat].where(df[feat].isin(set(lbe.classes_)), sparse_missing_val)
            df[feat] = lbe.transform(df[feat])

        df[dense_features] = mms.transform(df[dense_features])

        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=None, index=False)
        else:
            df.to_csv(output_path, index=False)


def preprosess_crieto_data(base_path, input_path, output_path):
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    cols = dense_features + sparse_features

    from collections import defaultdict
    sparse_missing_val = '-1'
    sparce_unique_vals = defaultdict(set)
    mms = MinMaxScaler(feature_range=(0, 1))
    label_encoders = []

    count = 0
    for df in pd.read_csv('train.txt', sep='\t', chunksize=100000, header=None, names=['label'] + cols):
        count += len(df)
        print(f'\rload {count}', end='')
        df[sparse_features] = df[sparse_features].fillna(sparse_missing_val)
        df[dense_features] = df[dense_features].fillna(0)
        for feat in sparse_features:
            sparce_unique_vals[feat].update(np.unique(df[feat]))
        mms.partial_fit(df[dense_features])
    for feat in sparse_features:
        lbe = LabelEncoder().fit(np.append(list(sparce_unique_vals[feat]), [sparse_missing_val]))
        label_encoders.append(lbe)
    with open('mms.pkl', 'wb') as f:
        pickle.dump(mms, f)
    with open('lbl_encoders.pkl', 'wb') as f:
        pickle.dump(len(label_encoders), f)
        for value in label_encoders:
            pickle.dump(value, f)

    ########### preprocess train ###############
    preprocess_data('train.txt', 'train_df_preprocessed.csv', sparse_features, dense_features, mms, sparse_missing_val)

    ########## preprocess test ###############
    preprocess_data(base_path / 'test.txt', base_path / 'test_preprocessed.csv', sparse_features, dense_features, mms,
                    sparse_missing_val)


def generate_arrays_from_file_criteo(batch_size):
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    while True:
        # data = pd.read_csv('train.txt',delimiter='\t', header=None, chunksize=batch_size,names=['label'] + cols)
        data = pd.read_csv('train_df_preprocessed.csv', chunksize=batch_size)
        for chunk in data:
            chunk[sparse_features] = chunk[sparse_features].fillna('-1')
            chunk[dense_features] = chunk[dense_features].fillna(0)

            for i, feat in enumerate(sparse_features):
                lbe = label_encoders[i]
                chunk[feat] = chunk[feat].where(chunk[feat].isin(set(lbe.classes_)), '-1')
                chunk[feat] = lbe.transform(chunk[feat])

                chunk[dense_features] = mms.transform(chunk[dense_features])
            y = chunk['label'].values
            train_model_input = {name: chunk[name].values for name in feature_names}
            yield (train_model_input, y)

