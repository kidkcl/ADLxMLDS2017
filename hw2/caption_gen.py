import tensorflow as tf
import keras
import numpy as np
import json
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Multiply, Activation, TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import pandas as pd

lantent_dim = 256

class Caption_Gen:
    
    def __init__(self):
        self.file_list = []
        self.labels = []
        self.vocab_size = 0
        self.max_caption_len = 40
        self.word2idx = {}
        self.idx2word = {}

    def read_data(self, feature_dir, label_path):
        # reading label json file
        with open(label_path) as label_json:
            raw_labels = json.load(label_json)

        label_dict = {}
        # self.file_list = list()
        training_data = np.zeros((len(raw_labels), 80, 4096))
        for idx in range(len(raw_labels)):
            key = raw_labels[idx]['id']
            self.file_list.append(raw_labels[idx]['id'])
            label_dict[key] = list(raw_labels[idx]['caption'])
            feat_path = feature_dir + key + '.npy'
            training_data[idx] = np.load(feat_path)
            
        # mark all labels with <bos> <eos>
        for key in label_dict:
            captions = label_dict[key]
            for i in range(len(captions)):
                captions[i] = '<bos> ' + captions[i] + ' <eos>'
            label_dict[key] = captions
            
        return label_dict, training_data
    
    def create_vocab(self, label_dict, n):
        print('creating vacabulary dict...')
        self.labels = []
        new_label = []
        for instance in self.file_list:
            captions = label_dict[instance]
            translation_table = dict.fromkeys(map(ord, '''1234567890,.~?!@#$&*+=<>'"][}{)(/'''), None)
            for j in range(len(captions)):
                new_label.append(captions[j].lower().translate(translation_table).split(' '))
            i = 0
            if n >= len(captions):
                i = len(captions) - 1
            else:
                i = n
            #print(len(captions), ' ', i)
            self.labels.append(captions[i].lower().translate(translation_table).split(' '))
        
        idx = 0
        # count = 0
        tmp_dict = {}
        for caption in new_label:
            for word in caption:
                if word not in tmp_dict:
                    tmp_dict[word] = 1
                else:
                    val = tmp_dict[word]
                    tmp_dict[word] = val + 1
        sorted_keys = sorted(tmp_dict.keys())
        for i in range(len(sorted_keys)):
            key = sorted_keys[i]
            if tmp_dict[key] > 1:
                self.word2idx[key] = idx
                self.idx2word[idx] = key
                idx += 1
        self.vocab_size = len(self.word2idx)
        #import ipdb; ipdb.set_trace()
        return self.labels
    
    def one_hot(self):
        print('one hot encoding...')
        z = self.vocab_size + 1
        encoded_tensor = np.zeros((len(self.labels), 121, z), dtype=np.float16)
        label_tensor = np.zeros((len(self.labels), 121, z), dtype =np.float16) # 121 the magic number...
        print(encoded_tensor.shape)
        #one-hot-encoding
        for i in range(len(self.labels)):
            # mark <BOS> to encoded_tensor, <EOS> to label_tensor
            encoded_tensor[i,80, self.word2idx['bos']] = 1
            label_tensor[i, 80 + len(self.labels[i]) - 2, self.word2idx['eos']] = 1
            for j in range(1, len(self.labels[i])-1):
                if self.labels[i][j] not in self.word2idx.keys(): 
                    encoded_tensor[i, j + 80, self.vocab_size] = 1
                    label_tensor[i, j + 79, self.vocab_size] = 1
                else:
                    k = self.labels[i][j]
                    encoded_tensor[i, j + 80, self.word2idx[k]] = 1
                    label_tensor[i, j + 79, self.word2idx[k]] = 1
                #if self.labels[i][j] not in self.word2idx.keys(): label_tensor[i, j+80+1, self.vocab_size] = 1
                #else:
                #    k = self.labels[i][j]
                #    label_tensor[i, j+80+1, self.word2idx[k]] = 1
                
        return encoded_tensor, label_tensor
    
    def padding_features(self, training_data):
        # pad videos fetures from 80 to 121
        new_feat = np.zeros([len(self.file_list), 121, 4096]) # 80 frames + <bos>/<eos> + max_caption_len
        for i in range(len(self.file_list)):
            for j in range(len(training_data[i])):
                new_feat[i][j] = training_data[i][j]
        return new_feat
    
    def build_model(self):
        input1 = Input(shape=(121, 4096))
        encoder_outputs = LSTM(lantent_dim, return_sequences=True)(input1)
        #encoder_outputs = encoder(input1)
        # concat encoder input with padding and captions,
        input2 = Input(shape=(121, self.vocab_size+1))
        ### decoding stage ??? ###
        encoder_pad = Concatenate(axis=2)([encoder_outputs, input2])
        decoder = LSTM(lantent_dim, return_sequences=True)
        decoder_outputs = decoder(encoder_pad)
        decoder_dense = TimeDistributed(Dense(self.vocab_size+1, activation=None))(decoder_outputs)
        input3 = Input(shape=(121, self.vocab_size+1))
        decoder_multiply = Multiply()([decoder_dense, input3])
        decoder_final = Activation('softmax')
        decoder_outputs = decoder_final(decoder_multiply)
        
        model = Model([input1, input2, input3], decoder_outputs)
        model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr=0.001))

        print(model.summary())
        return model
    
test = Caption_Gen()
l, t = test.read_data('MLDS_hw2_data/training_data/feat/', 'MLDS_hw2_data/training_label.json')
nl = test.create_vocab(l, 0)
print(nl[1])
et, lt = test.one_hot()
print('encoder tensor: ', et.shape)
nf = test.padding_features(t)
model = test.build_model()
#import ipdb; ipdb.set_trace()
hack_mat = []
for i in range(len(test.file_list)):
    tmp = []
    for j in range(121):
        if j < 80: tmp.append(np.zeros(test.vocab_size+1))
        else: tmp.append(np.ones(test.vocab_size+1))
    hack_mat.append(tmp)
hack_mat = np.array(hack_mat)
#model = keras.models.load_model('test_1113.h5')
#encoded_tensor = np.zeros((len(test.file_list), 121, test.vocab_size+1), dtype=np.float16)
#output = model.predict([nf, encoded_tensor, hack_mat])
#import ipdb; ipdb.set_trace()
#predict_captinos = []
#for i in range(1450):
#    text = ''
#    for j in range(80,121):
#        word = np.argmax(output[i,j,:])
#        if word == test.vocab_size: text = '???'
#        else: text += test.idx2word[word]
#        text += ' '
#    predict_captinos.append(text)
#pd.DataFrame({"caption": predict_captinos, "id": test.file_list}).to_csv('test.csv', index=False, header=True)
###
n_batches = 37 # max len of captions in one video
batches = np.arange(n_batches)
np.random.shuffle(batches)
for epoch in range(20):
    print('\nEpoch ', epoch+1)
    filepath = 'test_dim256/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    print('encoder tensor: ', et.shape)
    model.fit(x = [nf, et, hack_mat], y = lt, batch_size = 32, epochs= 5, callbacks = callbacks_list)
    nl = test.create_vocab(l, batches[epoch])
    print(nl[1])
    et, lt = test.one_hot()
#save final model(?
model.save('s2vt_dim256.h5')
#model= keras.models.load_model('ireallydontknow.h5')
#encoded_tensor = np.zeros((len(test.file_list), 121, test.vocab_size+1), dtype=np.float16)
#encoded_tensor[:,80,test.word2idx['bos']] = 1
#output = model.predict([nf, encoded_tensor, hack_mat])
#predict_captinos = []
#for i in range(1450):
#    text = ''
#    for j in range(80,121):
#        word = np.argmax(output[i,j,:])
#        if word == test.vocab_size: text = '???'
#        else: text += test.idx2word[word]
#        text += ' '
#    predict_captinos.append(text)
#print('===')
##import ipdb; ipdb.set_trace()
#pd.DataFrame({"id": test.file_list, "caption": predict_captinos}).to_csv('test.csv', index=False, header=True)
