import sys

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM

max_features = 69
batch_size = 32
epochs = 50
nb_class = 40

def load_mfcc_train_data(path):
    dim = 70
    features = []
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        print(len(lines))
        for i in lines:
            #import ipdb; ipdb.set_trace()
            features.append(i.split(" "))
        # print(features)
    return np.asarray(features).reshape((len(lines), dim))

def load_labels(path, map_path):
    labels = {}
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        print(len(lines))
        for i in lines:
            pair = i.split(',')
            labels[pair[0]] = pair[1]
    lab_dict = {}
    with open(map_path, 'r') as m:
        lines = m.read().splitlines()
        for i in lines:
            tmp = i.split('\t')
            lab_dict[tmp[0]] = tmp[1]
    s = set(val for val in lab_dict.values())
    #print(s)
    sorted_label = sorted(s)
    lb_to_class = dict((c, i) for i, c in enumerate(sorted_label))
    #print(lb_to_class)
    class_to_lb = dict((i, c) for i, c in enumerate(sorted_label))
    for key in labels:
        lab_key = labels[key]
        labels[key] = lab_dict[lab_key]
    return labels, lb_to_class, class_to_lb

def align_and_slice(feature, label, lb_to_class):
    print('original feature shape: ', feature.shape)
    #print(feature[0][0])
    new_label = {}
    # align label to train.ark
    for idx in range(len(feature)):
        instanceID = feature[idx][0]
        phone = label[instanceID]
        lb_idx = lb_to_class[phone]
        tokens = instanceID.split('_')
        instance = '_'.join(tokens[:2])
        if instance in new_label:
            new_label[instance].append(lb_idx)
        else:
            new_label[instance] = [lb_idx]
    for key in new_label:
        label_list = new_label[key]
        #sil = lb_to_class['sil']
        if len(label_list) < 777:
            label_list = label_list + [39]*(777-len(label_list))
            new_label[key] = label_list
    #import ipdb; ipdb.set_trace()
    # classify each instance
    inst_dict = {}
    for idx in range(len(feature)):
        instanceID = feature[idx][0]
        QQ = instanceID.split('_')
        instance = '_'.join(QQ[:2])
        if instance in inst_dict:
            inst_dict[instance].append(feature[idx][1:])
        else:
            inst_dict[instance] = []
            inst_dict[instance].append(feature[idx][1:])
    df = pd.DataFrame(dict((k, pd.Series(v)) for k,v in inst_dict.items()))
    pure_data = []
    tmp_label = []
    for col in df.columns[:]:
        df[col].loc[df[col].isnull()] = df[col].loc[df[col].isnull()].apply(lambda x: np.zeros(max_features))
        tmp = df[col].tolist()
        #import ipdb; ipdb.set_trace()
        pure_data.append(tmp)
        tmp_label.append(new_label[col])
    new_train = np.asarray(pure_data).astype(float)
    #import ipdb; ipdb.set_trace()
    print(new_train.shape)
    #tmp_label = np.array([list(s) for s in new_label.values()])
    tmp_label = np.asarray(tmp_label)
    print(tmp_label.shape)
    #import ipdb; ipdb.set_trace()
    ###
    #new_new_label = keras.utils.to_categorical(new_new_label, nb_class)
    new_new_label = []
    for i in range(len(tmp_label)):
        new_new_label.append(keras.utils.to_categorical(tmp_label[i], nb_class))
    #print(new_new_label.shape)
    new_new_label = np.asarray(new_new_label)
    new_new_label = new_new_label.astype(float)
    print('after slice: ', new_train.shape, new_new_label.shape)
    #print(new_train[0])
    return new_train, new_new_label

def normalize_data(data):
    for t in range(len(data)):
        for s in range(len(data[t])):
            mean = np.mean(data[t][s])
            std = np.std(data[t][s])
            if np.count_nonzero(data[t][s]) == 0: break
            for idx in range(len(data[t][s])):
                data[t][s][idx] = (data[t][s][idx] - mean) / std
    #print(data[0][0])
    return data

def build_rnn_model(train, label):
    model = Sequential()

    #model.add(TimeDistributed(Dense(max_features), input_shape=(777,69)))
    #model.add(Dropout(0.25))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(777,69)))
    model.add(TimeDistributed(Dense(nb_class, activation='softmax')))

    optimizer = keras.optimizers.RMSprop(lr=0.001, epsilon=1e-08, decay=0.0)
    #optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2, verbose=0, mode='auto') # for 300 epochs exp
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    model.fit(train, label,
              batch_size=batch_size, epochs=epochs,
              verbose=1, validation_split=0.1)

    return model

def run():
    root_dir = sys.argv[1]
    print('Reading training data...')
    train_path = root_dir + 'fbank/train.ark'
    feature = load_mfcc_train_data(train_path)
    print('Reading label file...')
    label_path = root_dir + 'train.lab'
    48_39_path = root_dir + 'phones/48_39.map'
    raw_label, lb_to_class, class_to_lb = load_labels(label_path, 48_39_path)
    train, label = align_and_slice(feature, raw_label, lb_to_class)
    #import ipdb; ipdb.set_trace()
    print('Normalizing...')
    train = normalize_data(train)
    #print(train[0][12])
    #print(train[0][21])
    #print(label[0][12])
    #print(label[0][21])
    print('Building model...')
    model = build_rnn_model(train, label)
    #model = keras.models.load_model('norm_model_rnn_rmrmsprop.h5')
    model.summary()
    score = model.evaluate(train, label, verbose=0)
    print('Test loss:', score)
    #model.save('norm_model_rnn_rmrmsprop.h5')

if __name__ == '__main__':
    run()
