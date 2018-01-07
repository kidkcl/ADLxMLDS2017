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

import model
from utils import image_processing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=100,\
            help='Noise dimension')
    parser.add_argument('--t_dim', type=int, default=256,\
            help='Text feature dimension')
    parser.add_argument('--batch_size', type=int, default=64,\
            help='Batch Size')
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
    parser.add_argument('--data_dir', type=str, default="data/",\
            help='Data Directory')
    parser.add_argument('--learning_rate', type=float, default=0.0002,\
            help='Learning Rate')
    parser.add_argument('--beta1', type=float, default=0.5,\
            help='Momentum for Adam Update')
    parser.add_argument('--epochs', type=int, default=200,\
            help='Max number of epochs')
    parser.add_argument('--save', type=str, default='models/',\
            help='Dir. to save the model(s)')
    parser.add_argument('--save_every', type=int, default=70,\
            help='Save Model/Samples every x iterations over batches')
    parser.add_argument('--resume_model', type=str, default=None,\
            help='Pre-Trained Model Path, to resume from')
    parser.add_argument('--seed', type=int, default=55668,\
            help='Fixed seed for reporducing')
    args = parser.parse_args()
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    model_options = {
            'z_dim' : args.z_dim,
            't_dim' : args.t_dim,
            'batch_size' : args.batch_size,
            'image_size' : args.image_size,
            'gf_dim' : args.gf_dim,
            'df_dim' : args.df_dim,
            'gfc_dim' : args.gfc_dim,
            'caption_vector_length' : args.caption_vector_length
    }

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        gan = model.GAN(model_options)
        input_tensors, variables, loss, outputs, checks = gan.build_model()
    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], \
            var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], \
            var_list=variables['g_vars'])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    if args.resume_model:
        print("Resume model from {}".format(args.resume_model))
        saver.restore(sess, tf.train.latest_checkpoint(args.resume_model))

    loaded_data = load_training_data(args.data_dir)

    for i in range(args.epochs):
        batch_no = 0
        start_time = time.time()
        while batch_no * args.batch_size < loaded_data['data_length']:
            real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no, args.batch_size, args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, loaded_data)

            check_ts = [checks['d_loss1'], checks['d_loss2'], checks['d_loss3']]

            _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,\
                    feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_real_caption'] : caption_vectors,
                            input_tensors['t_z'] : z_noise
                    })

            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],\
                    feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_real_caption'] : caption_vectors,
                            input_tensors['t_z'] : z_noise
                    })
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],\
                    feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_real_caption'] : caption_vectors,
                            input_tensors['t_z'] : z_noise
                    })
            if batch_no % 10 == 0:
                print("epoch: ", i, " | batch_no: [", batch_no, "/", len(loaded_data['image_list'])//args.batch_size, "] | ",\
                        "d_loss: ", d_loss, " | g_loss: ", g_loss, \
                        " | usage ", time.time() - start_time, " seconds.")
                print("d1: ", d1, " | d2: ", d2, " | d3: ", d3)
                start_time = time.time()
            batch_no += 1
            if (batch_no % args.save_every) == 0:
                print("Saving model")
                save_for_vis(real_images, gen, image_files)
                save_path = saver.save(sess, args.save + "/lastest_model.ckpt")

        if i % 5 == 0:
            save_path = saver.save(sess, args.save + "/model_epoch_{}.ckpt".format(i))

def load_training_data(data_dir):
    h = h5py.File(join(data_dir, 'faces.hdf5'))
    captions = {}
    try:
        for ds in h.iteritems():
            captions[ds[0]] = np.array(ds[1])
    except AttributeError: # Python 3
        for ds in h.items():
            captions[ds[0]] = np.array(ds[1])

    image_list = [img_name for img_name in captions]

    cut = 1
    img_cut = int(len(image_list)*cut)
    training_image_list = image_list[0:img_cut]
    random.shuffle(training_image_list)

    return {
            'image_list' : training_image_list,
            'captions' : captions,
            'data_length' : len(training_image_list)
    }

def save_for_vis(real_images, generated_images, image_files):
    if os.path.exists('samples'):
        shutil.rmtree('samples')
    os.makedirs('samples')

    for i in range(0, real_images.shape[0]):
        real_image_255 = np.zeros((64, 64, 3), dtype=np.uint8)
        real_images_255 = (real_images[i, :, :, :])
        scipy.misc.imsave('samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1]), real_images_255)
        fake_image_255 = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_images_255 = (generated_images[i, :, :, :])
        scipy.misc.imsave('samples/fake_image_{}.jpg'.format(i), fake_images_255)

def get_training_batch(batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, loaded_data=None):
    real_images = np.zeros((batch_size, 64, 64, 3))
    wrong_images = np.zeros((batch_size, 64, 64, 3))
    captions = np.zeros((batch_size, caption_vector_length))

    cnt = 0
    image_files = []

    for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
        idx = i % len(loaded_data['image_list'])
        image_file = join(data_dir, 'faces/' + loaded_data['image_list'][idx])
        image_array = image_processing.load_image_array(image_file, image_size)
        real_images[cnt, :, :, :] = image_array

        wrong_image_id = random.randint(0, len(loaded_data['image_list'])-1)
        wrong_image_file = join(data_dir, 'faces/' + loaded_data['image_list'][wrong_image_id])
        wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
        wrong_images[cnt, :, :, :] = wrong_image_array

        captions[cnt, :] = loaded_data['captions'][loaded_data['image_list'][idx]][0: caption_vector_length]
        image_files.append(image_file)
        cnt += 1
    
    z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
    return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
    main()
