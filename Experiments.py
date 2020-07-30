import pandas as pd
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM, xDeepFM, FGCNN
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.optimizers import Adam

from Layers_and_models import FGCNN_xDeepFM, FGCNN_ResxDeepFM


def prepare_data_for_train(train_sf, sparse_features, dense_features):
    data = train_sf
    # for sparse_feature in sparse_features:
    data[sparse_features] = data[sparse_features].fillna('-1')
    # for dense_feature in dense_features:
    data[dense_features] = data[dense_features].fillna(0)
    # target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    return data


def load_citero_dataset(data_path):
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    target = ['label']
    cols = target + dense_features + sparse_features
    train_df = pd.read_csv(data_path, delimiter='\t', header=None, names=cols)
    return train_df, sparse_features, dense_features, target


def load_taboola_dataset(data_path):
    train_df = pd.read_csv(data_path)
    target = ['is_click']
    dense_features = list(train_df.select_dtypes('number').drop(columns=target).columns)
    sparse_features = list(train_df.select_dtypes('object').columns)
    train_df[dense_features] = train_df[dense_features].astype(float)
    train_df[sparse_features] = train_df[sparse_features].astype(str)
    return train_df, sparse_features, dense_features, target


def deepfm_experiment(data_path, dataset_type='critero'):
    opt = Adam(lr=0.01)
    model_params = {'dnn_hidden_units': (400, 400),
                    'dnn_dropout': 0.5, 'dnn_activation': 'relu', 'task': 'binary'}
    run_base_experiment(data_path, dataset_type, model_params, DeepFM, opt)


def xdeepfm_experiment(data_path, dataset_type='critero'):
    opt = Adam(lr=0.01)
    model_params = {'dnn_hidden_units': (400, 400),
                    'dnn_dropout': 0.5, 'dnn_activation': 'relu', 'task': 'binary'}
    run_base_experiment(data_path, dataset_type, model_params, xDeepFM, opt)


def fgcnn_experiment(data_path, dataset_type='critero'):
    opt = Adam(lr=0.001)
    model_params = {
        'conv_kernel_width': (9, 9, 9, 9), 'conv_filters': (38, 40, 42, 44),
        'new_maps': (3, 3, 3, 3),
        'pooling_width': (1, 1, 1, 1), 'dnn_hidden_units': (4096, 2048, 1), 'l2_reg_linear': 1e-5,
        'l2_reg_embedding': 1e-5, 'l2_reg_dnn': 0,
        'dnn_dropout': 0, 'task': 'binary'
    }
    run_base_experiment(data_path, dataset_type, model_params, FGCNN, opt)


def fgcnn_xdeepfm_experiment(data_path, dataset_type='critero'):
    opt = Adam(lr=0.001)
    model_params = {
        'dnn_hidden_units': (256, 256),
        'cin_layer_size': (128, 128,), 'cin_split_half': True,
        'conv_kernel_width': (9, 9, 9, 9), 'conv_filters': (38, 40, 42, 44),
        'new_maps': (3, 3, 3, 3),
        'pooling_width': (1, 1, 1, 1), 'l2_reg_linear': 1e-5,
        'l2_reg_embedding': 1e-5, 'l2_reg_dnn': 0,
        'dnn_dropout': 0, 'task': 'binary'
    }
    run_base_experiment(data_path, dataset_type, model_params, FGCNN_xDeepFM, opt)


def fgnn_resxdeepfm(data_path, dataset_type='critero'):
    opt = Adam(lr=0.001)
    model_params = {
        'dnn_hidden_units': (256, 256),
        'cin_layer_size': (128, 128,), 'cin_split_half': True,
        'conv_kernel_width': (9, 9, 9, 9), 'conv_filters': (38, 40, 42, 44),
        'new_maps': (3, 3, 3, 3),
        'pooling_width': (1, 1, 1, 1), 'l2_reg_linear': 1e-5,
        'l2_reg_embedding': 1e-5, 'l2_reg_dnn': 0,
        'dnn_dropout': 0, 'task': 'binary', 'skip_rate':1
    }
    run_base_experiment(data_path, dataset_type, model_params, FGCNN_ResxDeepFM, opt)


def run_base_experiment(data_path, dataset_type, model_params, model_type, opt):
    if dataset_type == 'critero':
        data_df, sparse_features, dense_features, target = load_citero_dataset(data_path)
    else:
        data_df, sparse_features, dense_features, target = load_taboola_dataset(data_path)
    data_df = prepare_data_for_train(data_df, sparse_features, dense_features)
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data_df[feat].nunique(), embedding_dim=10)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    # 3.generate input data for model
    train, test = train_test_split(data_df, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    batch_size = 1024
    # 4.Define Model,train,predict and evaluate
    model = model_type(linear_feature_columns, dnn_feature_columns, seed=1024, **model_params)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=['binary_crossentropy', 'accuracy'], )
    history = model.fit(train_model_input, train[target].values,
                        batch_size=batch_size, epochs=10, verbose=1, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=batch_size)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    pass


if __name__ == "__main__":
    deepfm_experiment('citero_sample.csv', dataset_type='critero')
    # deepfm_experiment('taboola_sample.csv', dataset_type='tabbola')

    # xdeepfm_experiment('citero_sample.csv', dataset_type='critero')
    # xdeepfm_experiment('taboola_sample.csv', dataset_type='tabbola')

    # fgcnn_experiment('citero_sample.csv', dataset_type='critero')
    # fgcnn_experiment('taboola_sample.csv', dataset_type='tabbola')

    # fgcnn_xdeepfm_experiment('citero_sample.csv', dataset_type='critero')
    # fgcnn_xdeepfm_experiment('taboola_sample.csv', dataset_type='tabbola')

    # fgnn_resxdeepfm('citero_sample.csv', dataset_type='critero')
    # fgnn_resxdeepfm('taboola_sample.csv', dataset_type='tabbola')

