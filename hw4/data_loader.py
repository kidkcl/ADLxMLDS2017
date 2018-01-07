import json
import os
from os.path import join, isfile
import re
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import skipthoughts
import h5py

def get_caption_vectors(filename):
    captions_path = filename
    captions_pd = pd.read_csv(captions_path, header=None)

    model = skipthoughts.load_model()
    encoded_captions = {}

    ids = captions_pd[0]
    captions = captions_pd[1]
    print("Start to encode")
    st = time.time()
    encoded_captions_list = skipthoughts.encode(model, captions)
    print("All Seconds ", time.time() - st)

    return ids, encoded_captions_list

def save_caption_vectors(data_dir):
    captions_path = join(data_dir, 'tags.csv')
    captions_pd = pd.read_csv(captions_path, header=None)

    model = skipthoughts.load_model()
    encoded_captions = {}

    captions = captions_pd[1]
    print("Start to encode")
    st = time.time()
    encoded_captions_list = skipthoughts.encode(model, captions)
    print("All Seconds ", time.time() - st)

    for i, number in enumerate(captions_pd[0]):
        file_name = str(number) + ".jpg"
        encoded_captions[file_name] = encoded_captions_list[i]

    h = h5py.File(join(data_dir, 'faces.hdf5'))
    for key in encoded_captions:
        h.create_dataset(key, data=encoded_captions[key])
    h.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data',\
            help='Data directory')
    args = parser.parse_args()
    save_caption_vectors(args.data_dir)

if __name__ == '__main__':
    main()
