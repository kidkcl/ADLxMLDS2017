import sys

import numpy as np
import pandas as pd

import keras

max_features = 69

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

def load_label_map(path, char_path):
    lab_dict = {}
    with open(path, 'r') as m:
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
    char_map = {}
    with open(char_path, 'r') as f:
        lines = f.read().splitlines()
        for i in lines:
            group = i.split('\t')
            char_map[group[0]] = group[2]
    return lb_to_class, class_to_lb, char_map

def slice_data(data):
    print('original feature shape: ', data.shape)
    # classify each instance
    inst_dict = {}
    for idx in range(len(data)):
        instanceID = data[idx][0]
        QQ = instanceID.split('_')
        instance = '_'.join(QQ[:2])
        if instance in inst_dict:
            inst_dict[instance].append(data[idx][1:])
        else:
            inst_dict[instance] = []
            inst_dict[instance].append(data[idx][1:])
    df = pd.DataFrame(dict((k, pd.Series(v)) for k,v in inst_dict.items()))
    pure_data = []
    for col in df.columns[:]:
        df[col].loc[df[col].isnull()] = df[col].loc[df[col].isnull()].apply(lambda x: np.zeros(max_features))
        tmp = df[col].tolist()
        #import ipdb; ipdb.set_trace()
        pure_data.append(tmp)
    new_data = np.asarray(pure_data)
    b = np.zeros((len(new_data),777,max_features))
    b[:,:new_data.shape[1],:] = new_data
    #import ipdb; ipdb.set_trace()
    print(new_data.shape)
    b = b.astype(float)
    print('after slice: ', b.shape)
    #print(new_data[0])
    cols = df.columns[:]
    return b, cols, inst_dict

def normalize_data(data):
    for t in range(len(data)):
        for s in range(len(data[t])):
            mean = np.mean(data[t][s])
            std = np.std(data[t][s])
            if np.count_nonzero(data[t][s]) == 0: break
            for idx in range(len(data[t][s])):
                data[t][s][idx] = (data[t][s][idx] - mean) / std
    print(data[0][0])
    return data

def transfer_label(result, class_to_lb):
    predictions = []
    for i in range(len(result)):
        tmp = []
        for j in range(len(result[i])):
            tmp.append(class_to_lb[result[i][j]])
        predictions.append(tmp)
    predictions = np.asarray(predictions)
    print(predictions.shape)
    return predictions

def edit_distance(l_a, l_b):
    table = [[0] * (l_b + 1) for i in range(l_a+1)]
    for i in range(l_a + 1):
        for j in range(l_b + 1):
            if i == 0 and j == 0: continue
            table[i][j] = 999
            if i > 0: table[i][j] = min(table[i][j], table[i - 1][j] + 1)
            if j > 0: table[i][j] = min(table[i][j], table[i][j - 1] + 1)
            if i > 0 and j > 0:
                if s_a[i-1] == s_b[j-1]: table[i][j] = min(table[i][j], table[i - 1][j - 1])
                else: table[i][j] = min(table[i][j], table[i - 1][j - 1] + 1)
    return table[l_a][l_b]

def run():
    root_dir = sys.argv[1]
    out_name = sys.argv[2]
    model = keras.models.load_model('norm_model_rnn_adam.h5')
    test_path = root_dir + 'fbank/test.ark'
    raw_test = load_mfcc_train_data(test_path)
    new_test, instances, inst_dict = slice_data(raw_test)
    norm_test = normalize_data(new_test)
    classes = model.predict(norm_test)
    #import ipdb; ipdb.set_trace()
    #print(classes.shape)
    #classes = classes.astype(int)
    #print(classes[2])
    label_map_path = root_dir + '48_39.map'
    phone_path = root_dir + '48phone_char.map'
    lb_to_class, class_to_lb, char_map = load_label_map(label_map_path, phone_path)
    lb_to_class['null'] = 39
    class_to_lb[39] = 'null'
    #print(lb_to_class)
    #print(class_to_lb)
    tt = []
    for i in range(len(classes)):
        tmp_argmax = []
        for j in range(len(classes[i])):
            tmp = np.argmax(classes[i][j])
            tmp_argmax.append(tmp)
        tt.append(tmp_argmax)
    #tt = np.asarray(tt)
    #print(tt.shape)
    #print(tt)
    predictions = transfer_label(tt, class_to_lb)
    #import ipdb; ipdb.set_trace()
    answer = []
    for i in range(len(predictions)):
        res = predictions[i]
        key = instances[i]
        tmp = []
        last_seen = ''
        for i in range(len(res)-2):
            if len(tmp) > 0:
                last_seen = tmp[-1]
            if res[i] == res[i+1] == res[i+2] and res[i] != last_seen:
                tmp.append(res[i])
        tmp_str = ''
        for j in range(len(tmp)):
            if tmp[j] == 'null': continue
            tmp_str = tmp_str + char_map[tmp[j]]
        if tmp_str[0] == 'L' or tmp_str[-1] == 'L':
            tmp_str = tmp_str[1:-1]
        answer.append(tmp_str)
    pd.DataFrame({"id": instances, "phone_sequence": answer}).to_csv(out_name, index=False, header=True)

if __name__ == '__main__':
    run()
