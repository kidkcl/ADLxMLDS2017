from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import os
from os.path import join
import argparse
import pickle
import random
import json
import shutil
import time

import tensorflow as tf
import numpy as np
import scipy.misc
import h5py

import model as model
from utils import image_processing
from data_loader import get_caption_vectors

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_dim', type=int, default=100,\
            help='Noise dimension')
    parser.add_argument('--t_dim', type=int, default=256,\
            help='Text feature dimension')
    parser.add_argument('--image_size', type=int, default=64,\
            help='Image Size a, a x a')
    parser.add_argument('--gf_dim', type=int, default=64,\
            help='Number of conv in the first layer gen.')
    parser.add_argument('--df_dim', type=int, default=64,\
            help='Number of conv in the first layer discr.')
    parser.add_argument('--gfc_dim', type=int, default=1024,\
            help='Dimension of gen untis for for fully connected layer.')
    parser.add_argument('--caption_vector_length', type=int, default=4800,\
            help='Caption Vector Length')
    parser.add_argument('--model', type=str, default=None,\
            help='Path to the trained model')
    parser.add_argument('--caption_file', type=str, default=None,\
            help='Path to the testing caption file')
    parser.add_argument('--n_images', type=int, default=5,\
            help='Number of sampling image for per caption')
    args = parser.parse_args()

    model_options = {
            'z_dim': args.z_dim,
            't_dim': args.t_dim,
            'batch_size': args.n_images,
            'image_size': args.image_size,
            'gf_dim': args.gf_dim,
            'df_dim': args.df_dim,
            'gfc_dim': args.gfc_dim,
            'caption_vector_length': args.caption_vector_length
    }

    gan = model.GAN(model_options)
    _, _, _, _, _ = gan.build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    input_tensors, outputs = gan.build_generator()

    ids, captions_list = get_caption_vectors(args.caption_file)
    captions = np.array(captions_list)

    caption_image_dict = {}

    if os.path.exists('samples'):
        shutil.rmtree('samples')
    os.makedirs('samples')

    print('Sampling data')
    for cn, caption_vector in enumerate(captions):
        caption_images = []
        z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])
        #z_noise = np.ones([args.n_images, args.z_dim])
        print(z_noise)
        caption = [caption_vector[:args.caption_vector_length]] * args.n_images

        [gen_image] = sess.run([outputs['generator']],\
                feed_dict = {
                    input_tensors['t_real_caption']: caption,
                    input_tensors['t_z']: z_noise
                    }
                )
        for i in range(0, args.n_images):
            fake_image_255 = gen_image[i, :, :, :]
            scipy.misc.imsave('samples/sample_{}_{}.jpg'.format(ids[cn], i + 1), fake_image_255)


if __name__ == '__main__':
    main()
