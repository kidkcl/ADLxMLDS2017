import tensorflow as tf
import keras
import numpy as np
import json
import os
import pickle
import sys
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Multiply, Activation, TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import pandas as pd

#lantent_dim = 128

class Caption_Decode:
    
    def __init__(self):
        self.labels = []
        self.vocab_size = 0
        self.max_caption_len = 40
        self.word2idx = {}
        self.idx2word = {}

    def read_data(self, feature_dir, txt_path):
        # reading file id txt file
        file_list = []
        with open(txt_path) as f:
            for line in f.readlines():
                line = line.rstrip()
                file_list.append(line)

        # self.file_list = list()
        training_data = np.zeros((len(file_list), 80, 4096))
        for idx in range(len(file_list)):
            key = file_list[idx]
            feat_path = feature_dir + key + '.npy'
            training_data[idx] = np.load(feat_path)
            
        return training_data, file_list
    
    def read_vocabulary(self):
        self.word2idx = pickle.load(open("word2idx.pkl", "rb"))
        self.idx2word = pickle.load(open("idx2word.pkl", "rb"))
        self.vocab_size = len(self.word2idx)   
    
    def one_hot(self, result):
        print('one hot encoding...')
        z = self.vocab_size + 1
        encoded_tensor = np.zeros((len(result), 121, z), dtype=np.float16)
        label_tensor = np.zeros((len(result), 121, z), dtype =np.float16) # 121 the magic number...
        print(encoded_tensor.shape)
        #one-hot-encoding
        for i in range(len(result)):
            # mark <BOS> to encoded_tensor, <EOS> to label_tensor
            encoded_tensor[i,80, self.word2idx['bos']] = 1
            label_tensor[i, 80 + len(result[i]) - 2, self.word2idx['eos']] = 1
            for j in range(1, len(result[i])-1):
                if result[i][j] not in self.word2idx.keys(): 
                    encoded_tensor[i, j + 80, self.vocab_size] = 1
                    label_tensor[i, j + 79, self.vocab_size] = 1
                else:
                    k = result[i][j]
                    encoded_tensor[i, j + 80, self.word2idx[k]] = 1
                    label_tensor[i, j + 79, self.word2idx[k]] = 1
                
        return encoded_tensor, label_tensor
    
    def padding_features(self, training_data, file_list):
        # pad videos fetures from 80 to 121
        new_feat = np.zeros([len(file_list), 121, 4096]) # 80 frames + <bos>/<eos> + max_caption_len
        for i in range(len(file_list)):
            for j in range(len(training_data[i])):
                new_feat[i][j] = training_data[i][j]
        return new_feat

    
    def decode(self, output, file_list):
        predict_captinos = []
        for i in range(len(file_list)):
            text = ''
            for j in range(80,121):
                word = np.argmax(output[i,j,:])
                if word == self.vocab_size: text += '???'
                else: text += self.idx2word[word]
                text += ' '
            predict_captinos.append(text)
        return predict_captinos
    
    def my_predict(self, model, test_feature, encoded_tensor, mask_tensor):
        #stopped_condition = False
        caption = ''
        pos = 80
        while True:
            output = model.predict([test_feature, encoded_tensor, mask_tensor])
            word_id = np.argmax(output[0,pos,:])
            pos += 1
            if word_id == self.word2idx['eos'] or pos == 121:
                #stopped_condition = True
                break
            encoded_tensor[:,pos,word_id] = 1
            if word_id == self.vocab_size: caption += 'unknown'
            else: caption += self.idx2word[word_id]
            caption += ' '

        return caption

test = Caption_Decode()
root_dir = sys.argv[1]
feat_path = root_dir + 'testing_data/feat/'
id_path = root_dir + 'testing_id.txt'
testing_vids, file_list = test.read_data(feat_path, id_path) # fixme ...
test.read_vocabulary()

#et, lt = test.one_hot()
et = np.zeros((len(file_list),121,test.vocab_size+1))
et[:,80,test.word2idx['bos']] = 1
print('encoder tensor: ', et.shape)
nf = test.padding_features(testing_vids, file_list)
#model = test.build_model()
hack_mat = []
for i in range(len(file_list)):
    tmp = []
    for j in range(121):
        if j < 80: tmp.append(np.zeros(test.vocab_size+1))
        else: tmp.append(np.ones(test.vocab_size+1))
    hack_mat.append(tmp)
hack_mat = np.array(hack_mat)

model = keras.models.load_model('s2vt_dim256.h5')
#a = model.get_weights()
#print(len(a))
output = model.predict([nf, et, hack_mat])
#predict_captinos = test.decode(output, tlist)
#pd.DataFrame({"id": tlist, "captions": predict_captinos}).to_csv('test6.csv', index=False, header=True)
seq_predictions = []
for i in range(len(file_list)):
    each_nf = nf[i,:,:].reshape(1,121,4096)
    each_et = et[i,:,:].reshape(1,121,test.vocab_size+1)
    each_mt = hack_mat[i,:,:].reshape(1,121,test.vocab_size+1)
    seq = test.my_predict(model, each_nf, each_et, each_mt)
    seq_predictions.append(seq)

#import ipdb; ipdb.set_trace()
#tlist_short = tlist[:20]
#pd.DataFrame({"id": tlist_short, "captions": seq_predictions}).to_csv('test4.csv', index=False, header=True)
output_path = sys.argv[2]
with open(output_path, 'a') as out:
    for i in range(len(file_list)):
        out.write(file_list[i] + ',' + seq_predictions[i] + '\n')

peer = Caption_Decode()
feat_path = root_dir + 'peer_review/feat/'
id_path = root_dir + 'peer_review_id.txt'
testing_vids, file_list = peer.read_data(feat_path, id_path) # fixme ...
peer.read_vocabulary()

#et, lt = test.one_hot()
et = np.zeros((len(file_list),121,peer.vocab_size+1))
et[:,80,test.word2idx['bos']] = 1
print('encoder tensor: ', et.shape)
nf = peer.padding_features(testing_vids, file_list)
#model = test.build_model()
hack_mat = []
for i in range(len(file_list)):
    tmp = []
    for j in range(121):
        if j < 80: tmp.append(np.zeros(test.vocab_size+1))
        else: tmp.append(np.ones(test.vocab_size+1))
    hack_mat.append(tmp)
hack_mat = np.array(hack_mat)

output = model.predict([nf, et, hack_mat])
#predict_captinos = test.decode(output, tlist)
#pd.DataFrame({"id": tlist, "captions": predict_captinos}).to_csv('test6.csv', index=False, header=True)
seq_predictions = []
for i in range(len(file_list)):
    each_nf = nf[i,:,:].reshape(1,121,4096)
    each_et = et[i,:,:].reshape(1,121,test.vocab_size+1)
    each_mt = hack_mat[i,:,:].reshape(1,121,test.vocab_size+1)
    seq = peer.my_predict(model, each_nf, each_et, each_mt)
    seq_predictions.append(seq)

#import ipdb; ipdb.set_trace()
#tlist_short = tlist[:20]
#pd.DataFrame({"id": tlist_short, "captions": seq_predictions}).to_csv('test4.csv', index=False, header=True)
output_path = sys.argv[3]
with open(output_path, 'a') as out:
    for i in range(len(file_list)):
        out.write(file_list[i] + ',' + seq_predictions[i] + '\n')
