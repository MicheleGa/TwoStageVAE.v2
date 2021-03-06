import sys

sys.path.append('/content/drive/My Drive/VAE/VAE')
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.training.moving_averages import assign_moving_average
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from imageio import imread, imwrite
from scipy.ndimage import gaussian_filter
import random
import pickle
import time
from datetime import datetime
import fid

from DBSRCNN import DBSRCNN
from DBSRCNN import PSNRLoss


class BatchNorm(tf.keras.layers.Layer):

    def __init__(self, name):
        super(BatchNorm, self).__init__(name=name)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(shape=(dim,), initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=(dim,), initializer='zeros', trainable=True, name='beta')
        self.var = self.add_weight(shape=(dim,), initializer='ones', trainable=False, name='variance')
        self.mean = self.add_weight(shape=(dim,), initializer='zeros', trainable=False, name='mean')

    def call(self, inputs, training, scope, eps=1e-5, decay=0.999, affine=True, **kwargs):
        def mean_var_with_update(x, moving_mean, moving_variance):
            if len(x.get_shape().as_list()) == 4:
                statistics_axis = [0, 1, 2]
            else:
                statistics_axis = [0]
            mean, variance = tf.nn.moments(x, statistics_axis, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        with tf.name_scope(scope):
            mean, variance = tf.cond(training, lambda: mean_var_with_update(inputs, self.mean, self.var),
                                     lambda: (self.mean, self.var))
            if affine:
                return tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.gamma, eps)
            else:
                return tf.nn.batch_normalization(inputs, mean, variance, None, None, eps)


class DownSample(tf.keras.layers.Layer):

    def __init__(self, name, out_dim, kernel_size, l2_reg=None):
        super(DownSample, self).__init__(name=name)
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.l2_reg = l2_reg
        self.kernel_reg = regularizers.l2(self.l2_reg) if l2_reg is not None else None
        self.conv2DLayer = tf.keras.layers.Conv2D(out_dim, kernel_size, strides=2, padding='same',
                                                  kernel_regularizer=self.kernel_reg, name='DownSampleConv2D')

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape().as_list()
        assert (len(input_shape) == 4)
        return self.conv2DLayer(inputs)


class UpSample(tf.keras.layers.Layer):

    def __init__(self, name, out_dim, kernel_size, l2_reg=None):
        super(UpSample, self).__init__(name)
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.l2_reg = l2_reg
        self.kernel_reg = regularizers.l2(self.l2_reg) if l2_reg is not None else None
        self.conv2DTransposeLayer = tf.keras.layers.Conv2DTranspose(out_dim, kernel_size, strides=2, padding='same',
                                                                    kernel_regularizer=self.kernel_reg,
                                                                    name='UpSampleConv2D')

    def call(self, inputs, **kwargs):
        input_shape = inputs.get_shape().as_list()
        assert (len(input_shape) == 4)
        return self.conv2DTransposeLayer(inputs)


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, name, out_dim, depth=2, kernel_size=3):
        super(ResBlock, self).__init__(name=name)
        self.out_dim = out_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.batchNormLayers = [BatchNorm('btn' + str(i)) for i in range(depth)]
        self.conv2DLayers = [tf.keras.layers.Conv2D(out_dim, kernel_size, padding='same', name='layer' + str(i),
                                                    data_format='channels_last', kernel_initializer=None)
                             for i in range(depth)]
        self.shortcutConv2DLayer = tf.keras.layers.Conv2D(out_dim, kernel_size, padding='same', name='shortcut',
                                                          data_format='channels_last', kernel_initializer=None)

    def call(self, inputs, is_training=None, **kwargs):
        y = inputs
        for i in range(self.depth):
            y = tf.nn.relu(self.batchNormLayers[i](y, is_training, 'bn' + str(i)))
            y = self.conv2DLayers[i](y)
        s = self.shortcutConv2DLayer(inputs)
        return y + s


class ResFcBlock(tf.keras.layers.Layer):

    def __init__(self, name, out_dim, depth=2):
        super(ResFcBlock, self).__init__(name=name)
        self.out_dim = out_dim
        self.depth = depth
        self.denseLayers = [tf.keras.layers.Dense(out_dim, name='layer' + str(i), kernel_initializer=None)
                            for i in range(depth)]
        self.shortcutDenseLayer = tf.keras.layers.Dense(out_dim, name='shortcut', kernel_initializer=None)

    def call(self, inputs, **kwargs):
        y = inputs
        for i in range(self.depth):
            y = self.denseLayers[i](tf.nn.relu(y))
        s = self.shortcutDenseLayer(inputs)
        return y + s


class ScaleBlock(tf.keras.layers.Layer):

    def __init__(self, name, out_dim, block_per_scale=1, depth_per_block=2, kernel_size=3):
        super(ScaleBlock, self).__init__(name=name)
        self.out_dim = out_dim
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.resBlocks = [ResBlock('ResBlock' + str(i), out_dim, depth_per_block, kernel_size)
                          for i in range(block_per_scale)]

    def call(self, inputs, is_training=None, **kwargs):
        y = inputs
        for i in range(self.block_per_scale):
            y = self.resBlocks[i](y, is_training)
        return y


class ScaleFcBlock(tf.keras.layers.Layer):

    def __init__(self, out_dim, block_per_scale=1, depth_per_block=2, name='ScaleFcBlock'):
        super(ScaleFcBlock, self).__init__(name=name)
        self.out_dim = out_dim
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.resFcBlocks = [ResFcBlock('block' + str(i), out_dim, depth_per_block) for i in range(block_per_scale)]

    def call(self, inputs, **kwargs):
        y = inputs
        for i in range(self.block_per_scale):
            y = self.resFcBlocks[i](y)
        return y


# Model

class Encoder1(tf.keras.layers.Layer):

    def __init__(self, base_dim, kernel_size, num_scale, block_per_scale, depth_per_block,
                 l2_reg, fc_dim, latent_dim, batch_size, name='Encoder1'):
        super(Encoder1, self).__init__(name=name)
        self.base_dim = base_dim
        self.kernel_size = kernel_size
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.l2_reg = l2_reg
        self.fc_dim = fc_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.conv0 = tf.keras.layers.Conv2D(base_dim, kernel_size, 1, padding='same', name='conv0',
                                            data_format='channels_last', kernel_initializer=None)
        self.scaleBlocks = [ScaleBlock('ScaleBlock' + str(i), base_dim, block_per_scale, depth_per_block, kernel_size)
                            for i in range(num_scale)]
        self.downSampleLayers = [DownSample('DownSample' + str(i), base_dim, kernel_size, l2_reg) for i in
                                 range(num_scale - 1)]
        self.scaleFcBlock = ScaleFcBlock(fc_dim, 1, depth_per_block)
        self.denseLayerMu_z = tf.keras.layers.Dense(latent_dim, kernel_initializer=None)
        self.denseLayerLogsd_z = tf.keras.layers.Dense(latent_dim, kernel_initializer=None)

    def call(self, inputs, is_training=None, **kwargs):
        dim = self.base_dim
        y = self.conv0(inputs)
        for i in range(self.num_scale):
            y = self.scaleBlocks[i](y, is_training)

            if i != self.num_scale - 1:
                dim *= 2
                y = self.downSampleLayers[i](y)

        y = tf.reduce_mean(y, [1, 2])
        y = self.scaleFcBlock(y)

        mu_z = self.denseLayerMu_z(y)
        logsd_z = self.denseLayerLogsd_z(y)
        sd_z = tf.exp(logsd_z)
        z = mu_z + tf.random.normal([self.batch_size, self.latent_dim]) * sd_z
        return mu_z, logsd_z, sd_z, z


class Decoder1(tf.keras.layers.Layer):

    def __init__(self, inp_shape, dims, scales, kernel_size, l2_reg, block_per_scale, depth_per_block, name='Decoder1'):
        super(Decoder1, self).__init__(name=name)
        self.dims = dims
        self.scales = scales
        self.kernel_size = kernel_size
        self.l2_reg = l2_reg
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.data_depth = inp_shape[-1]
        self.fc_dim = 2 * 2 * dims[0]
        self.denseLayer = tf.keras.layers.Dense(self.fc_dim, name='fc0', kernel_initializer=None)
        self.upSampleLayers = [UpSample('UpSample' + str(i), dims[i + 1], kernel_size, l2_reg) for i in
                               range(len(scales) - 1)]
        self.scaleBlocks = [
            ScaleBlock('ScaleBlock' + str(i), dims[i + 1], block_per_scale, depth_per_block, kernel_size)
            for i in range(len(scales) - 1)]
        self.conv2DLayer = tf.keras.layers.Conv2D(self.data_depth, kernel_size, 1, padding='same',
                                                  data_format='channels_last', kernel_initializer=None)

    def call(self, inputs, is_training=None, **kwargs):
        y = inputs
        y = self.denseLayer(y)
        y = tf.reshape(y, [-1, 2, 2, self.dims[0]])

        for i in range(len(self.scales) - 1):
            y = self.upSampleLayers[i](y)
            y = self.scaleBlocks[i](y, is_training)

        y = self.conv2DLayer(y)
        x_hat = tf.nn.sigmoid(y)
        return x_hat


class Encoder2(tf.keras.layers.Layer):

    def __init__(self, second_depth, second_dim, latent_dim, batch_size, name='Encoder2'):
        super(Encoder2, self).__init__(name=name)
        self.second_depth = second_depth
        self.second_dim = second_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.denseLayers = [tf.keras.layers.Dense(second_dim, activation=tf.nn.relu, name='fc' + str(i),
                                                  kernel_initializer=None)
                            for i in range(second_depth)]
        self.denseLayerMu_u = tf.keras.layers.Dense(latent_dim, name='mu_u', kernel_initializer=None)
        self.denseLayerLogsd_u = tf.keras.layers.Dense(latent_dim, name='logsd_u', kernel_initializer=None)

    def call(self, inputs, **kwargs):
        t = inputs
        for i in range(self.second_depth):
            t = self.denseLayers[i](t)
        t = tf.concat([inputs, t], -1)

        mu_u = self.denseLayerMu_u(t)
        logsd_u = self.denseLayerLogsd_u(t)
        sd_u = tf.exp(logsd_u)
        u = mu_u + sd_u * tf.random.normal([self.batch_size, self.latent_dim])
        return mu_u, logsd_u, sd_u, u


class Decoder2(tf.keras.layers.Layer):

    def __init__(self, second_depth, second_dim, latent_dim, name='Decoder2'):
        super(Decoder2, self).__init__(name=name)
        self.second_depth = second_depth
        self.second_dim = second_dim
        self.latent_dim = latent_dim
        self.denseLayers = [tf.keras.layers.Dense(second_dim, tf.nn.relu, name='fc' + str(i), kernel_initializer=None)
                            for i in range(second_depth)]
        self.denseLayerZ_hat = tf.keras.layers.Dense(latent_dim, name='z_hat', kernel_initializer=None)

    def call(self, inputs, **kwargs):
        t = inputs
        for i in range(self.second_depth):
            t = self.denseLayers[i](t)
        t = tf.concat([inputs, t], -1)
        z_hat = self.denseLayerZ_hat(t)
        return z_hat


class TwoStageVaeModel(tf.keras.Model):

    def __init__(self, inp_shape, latent_dim=64, second_depth=3, second_dim=1024, name='2StageVAE'):
        super(TwoStageVaeModel, self).__init__(name=name)
        self.inp_shape = inp_shape
        self.batch_size = inp_shape[0]
        self.latent_dim = latent_dim
        self.second_dim = second_dim
        self.img_dim = inp_shape[1]
        self.second_depth = second_depth

        self.__build_network()

    def __build_network(self):
        with tf.name_scope('stage1'):
            self.build_encoder1()
            self.build_decoder1()
        with tf.name_scope('stage2'):
            self.build_encoder2()
            self.build_decoder2()

    def compute_loss1(self, x, x_hat, mse):
        HALF_LOG_TWO_PI = 0.91893
        k = (2 * self.img_dim / self.latent_dim) ** 2
        self.kl_loss1 = tf.math.reduce_sum(
            tf.math.square(self.mu_z) + tf.math.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size)
        self.mse_loss1 = mse(x, x_hat)
        self.loggamma_x = tf.math.log(self.gamma_x)
        self.gen_loss1 = tf.math.reduce_sum(tf.math.square((x - x_hat) / self.gamma_x) / 2.0 + self.loggamma_x +
                                            HALF_LOG_TWO_PI) / float(self.batch_size)

        self.loss1 = k * self.kl_loss1 + self.gen_loss1
        return self.loss1, self.mse_loss1

    def compute_loss2(self, z, z_hat, mse):
        HALF_LOG_TWO_PI = 0.91893
        self.loggamma_z = tf.math.log(self.gamma_z)
        self.kl_loss2 = tf.math.reduce_sum(
            tf.math.square(self.mu_u) + tf.math.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.mse_loss2 = mse(z, z_hat)
        self.gen_loss2 = tf.reduce_sum(tf.math.square((z - z_hat) / self.gamma_z) / 2.0 + self.loggamma_z +
                                       HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2
        return self.loss2, self.mse_loss2

    def build_encoder2(self):
        self.encoder2 = Encoder2(self.second_depth, self.second_dim, self.latent_dim, self.batch_size)

    def build_decoder2(self):
        self.decoder2 = Decoder2(self.second_depth, self.second_dim, self.latent_dim)

    def extract_posterior(self, x):
        num_sample = np.shape(x)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        mu_z, sd_z = [], []
        for i in range(num_iter):
            inputs = tf.cast(x_extend[i * self.batch_size:(i + 1) * self.batch_size], dtype=tf.float32) / 255.
            mu_z_batch, _, sd_z_batch, _ = self.encoder1(inputs=inputs, is_training=False)
            mu_z.append(mu_z_batch)
            sd_z.append(sd_z_batch)
        mu_z = np.concatenate(mu_z, 0)[0:num_sample]
        sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        return mu_z, sd_z

    def step(self, stage, step, input_batch, gamma, lr, checkpoint, writer=None, write_iteration=600):
        self.lr = lr
        mse = tf.keras.losses.MeanSquaredError()
        x = tf.cast(input_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            if stage == 1:
                x = x / 255.
                self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-08)
                self.gamma_x = gamma
                self.mu_z, self.logsd_z, self.sd_z, self.z = self.encoder1(inputs=x, is_training=True)
                x_hat = self.decoder1(inputs=self.z, is_training=True)
                loss, mse_loss = self.compute_loss1(x, x_hat, mse)
                optimizer = self.optimizer1
            elif stage == 2:
                self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-08)
                self.gamma_z = gamma
                self.mu_u, self.logsd_u, self.sd_u, self.u = self.encoder2(inputs=x)
                z_hat = self.decoder2(inputs=self.u)
                loss, mse_loss = self.compute_loss2(x, z_hat, mse)
                optimizer = self.optimizer2
            else:
                raise Exception('Wrong stage {}.'.format(stage))

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if step % write_iteration == 0 and writer is not None:
            with writer.as_default():
                if stage == 1:
                    tf.summary.scalar('gamma_x', self.gamma_x, step=step)
                    tf.summary.scalar('stage1_loss', loss, step=step)
                    tf.summary.scalar('stage1_mse_loss', mse_loss, step=step)
                elif stage == 2:
                    tf.summary.scalar('gamma_z', self.gamma_z, step=step)
                    tf.summary.scalar('stage2_loss', loss, step=step)
                    tf.summary.scalar('stage2_mse_loss', mse_loss, step=step)
                else:
                    raise Exception('Wrong stage {}.'.format(stage))

        checkpoint.step.assign_add(1)
        return loss, mse_loss

    def reconstruct2(self, z):
        # reconstruction of latent space by the second stage
        num_sample = np.shape(z)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        z_extend = np.concatenate([z, z[0:self.batch_size]], 0)
        recon_z = []
        for i in range(num_iter):
            inputs = tf.cast(z_extend[i * self.batch_size:(i + 1) * self.batch_size], dtype=tf.float32)
            mu_u, logsd_u, sd_u, u = self.encoder2(inputs=inputs)
            recon_z_batch = self.decoder2(inputs=u)
            recon_z.append(recon_z_batch)
        recon_z = np.concatenate(recon_z, 0)[0:num_sample]
        return recon_z

    def reconstruct(self, x):
        num_sample = np.shape(x)[0]
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        recon_x = []
        mu_z_tot = []
        logsd_z_tot = []
        for i in range(num_iter):
            # Get mu_z and logsd_z for every batch of data

            inputs = tf.cast(x_extend[i * self.batch_size:(i + 1) * self.batch_size], dtype=tf.float32) / 255.
            mu_z_batch, logsd_z_batch, sd_z_batch, z_batch = self.encoder1(inputs=inputs, is_training=False)
            recon_x_batch = self.decoder1(inputs=z_batch, is_training=False)

            recon_x.append(recon_x_batch)
            mu_z_tot.append(mu_z_batch)
            logsd_z_tot.append(logsd_z_batch)
        recon_x = np.concatenate(recon_x, 0)[0:num_sample]
        mu_z_tot = np.concatenate(mu_z_tot, 0)[0:num_sample]
        logsd_z_tot = np.concatenate(logsd_z_tot, 0)[0:num_sample]
        # Return recon_x
        return mu_z_tot, logsd_z_tot, recon_x

    def generate(self, num_sample, stage=2, adjust2=None, adjust1=None):
        num_iter = int(math.ceil(float(num_sample) / float(self.batch_size)))
        gen_samples = []
        gen_z = []
        for i in range(num_iter):
            if stage == 2:
                # u ~ N(0, I)
                u = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                # z ~ N(f_2(u), \gamma_z I)
                z = self.decoder2(inputs=tf.cast(u, dtype=tf.float32))
                if type(adjust2) == np.float32:  # np.ndarray
                    # print("normalizing 2")
                    rescale = adjust2 / np.mean(np.std(z, axis=0))
                    # print(rescale)
                    z = z * rescale
            else:
                z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                if type(adjust2) == np.float32:  # np.ndarray
                    z = z * adjust2
            # x = f_1(z)
            x = self.decoder1(inputs=tf.cast(z, dtype=tf.float32), is_training=False)
            gen_z.append(z)
            gen_samples.append(x)
        gen_z = np.concatenate(gen_z, 0)
        gen_samples = np.concatenate(gen_samples, 0)
        if type(adjust1) == np.float:
            rescale = adjust1 / np.mean(np.std(gen_samples, axis=0))
            gmean = np.mean(gen_samples, axis=0)
            gen_samples = (gen_samples - gmean) * rescale + gmean
            # Need to remain in range 0-1
            gen_samples = np.maximum(np.minimum(gen_samples, 1), 0)
        return gen_samples[0:num_sample], gen_z[0:num_sample]


class Resnet(TwoStageVaeModel):
    def __init__(self, inp_shape, num_scale, block_per_scale=1, depth_per_block=2, kernel_size=3, base_dim=16,
                 fc_dim=512, latent_dim=64, second_depth=3, second_dim=1024, l2_reg=.001):
        self.num_scale = num_scale
        self.block_per_scale = block_per_scale
        self.depth_per_block = depth_per_block
        self.kernel_size = kernel_size
        self.base_dim = base_dim
        self.fc_dim = fc_dim
        self.l2_reg = l2_reg
        super(Resnet, self).__init__(inp_shape, latent_dim, second_depth, second_dim)

    def build_encoder1(self):
        self.encoder1 = Encoder1(self.base_dim, self.kernel_size, self.num_scale,
                                 self.block_per_scale, self.depth_per_block, self.l2_reg, self.fc_dim,
                                 self.latent_dim, self.batch_size)

    def build_decoder1(self):
        desired_scale = self.inp_shape[1]
        scales, dims = [], []
        current_scale, current_dim = 2, self.base_dim
        while current_scale <= desired_scale:
            scales.append(current_scale)
            dims.append(current_dim)
            current_scale *= 2
            current_dim = min(current_dim * 2, 1024)
        assert (scales[-1] == desired_scale)
        dims = list(reversed(dims))

        self.decoder1 = Decoder1(self.inp_shape, dims, scales, self.kernel_size, self.l2_reg,
                                 self.block_per_scale, self.depth_per_block)


def load_cifar10_data(flag='training'):
    if flag == 'training':
        data_files = ['data/cifar10/cifar-10-batches-py/data_batch_1', 'data/cifar10/cifar-10-batches-py/data_batch_2',
                      'data/cifar10/cifar-10-batches-py/data_batch_3', 'data/cifar10/cifar-10-batches-py/data_batch_4',
                      'data/cifar10/cifar-10-batches-py/data_batch_5']
    else:
        data_files = ['data/cifar10/cifar-10-batches-py/test_batch']
    x = []
    y = []
    for filename in data_files:
        img_dict = unpickle(filename)
        img_data = img_dict[b'data']
        img_data = np.transpose(np.reshape(img_data, [-1, 3, 32, 32]), [0, 2, 3, 1])
        x.append(img_data)
    x = np.concatenate(x, 0)
    num_imgs = np.shape(x)[0]

    # save to jpg file
    img_folder = os.path.join('data/cifar10', flag)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    for i in range(num_imgs):
        imwrite(os.path.join(img_folder, str(i) + '.jpg'), x[i])

    # save to npy
    x = []
    for i in range(num_imgs):
        img_file = os.path.join(img_folder, str(i) + '.jpg')
        img = imread(img_file, pilmode='RGB')
        x.append(np.reshape(img, [1, 32, 32, 3]))
    x = np.concatenate(x, 0)

    return x.astype(np.uint8)


def load_celeba_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    imgs = []

    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[45:173, 25:153]
        img = np.array(Image.fromarray(img).resize((side_length, side_length), resample=Image.BILINEAR))
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img)
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


# Center crop 128x128 and resize to 64x64
def preprocess_celeba():
    x_val = load_celeba_data('val', 64)
    np.save(os.path.join('data', 'celeba', 'val.npy'), x_val)
    x_test = load_celeba_data('test', 64)
    np.save(os.path.join('data', 'celeba', 'test.npy'), x_test)
    x_train = load_celeba_data('training', 64)
    np.save(os.path.join('data', 'celeba', 'train.npy'), x_train)


def preporcess_cifar10():
    x_train = load_cifar10_data('training')
    np.save(os.path.join('data', 'cifar10', 'train.npy'), x_train)
    x_test = load_cifar10_data('testing')
    np.save(os.path.join('data', 'cifar10', 'test.npy'), x_test)


def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def load_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels


def load_test_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, 'data', name)
    if name.lower() == 'mnist' or name.lower() == 'fashion':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 32
        channels = 3
    elif name.lower() == 'celeba140':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels


def stich_imgs(x, img_per_row=10, img_per_col=10):
    x_shape = np.shape(x)
    assert (len(x_shape) == 4)
    output = np.zeros([img_per_row * x_shape[1], img_per_col * x_shape[2], x_shape[3]])
    idx = 0
    for r in range(img_per_row):
        start_row = r * x_shape[1]
        end_row = start_row + x_shape[1]
        for c in range(img_per_col):
            start_col = c * x_shape[2]
            end_col = start_col + x_shape[2]
            output[start_row:end_row, start_col:end_col] = x[idx]
            idx += 1
            if idx == x_shape[0]:
                break
        if idx == x_shape[0]:
            break
    if np.shape(output)[-1] == 1:
        output = np.reshape(output, np.shape(output)[0:2])
    return output


def stich_imgs_2(x_raw, x, img_per_row=10, img_per_col=2):
    x_shape = np.shape(x)
    assert (len(x_shape) == 4)
    output = np.zeros([img_per_col * x_shape[2], img_per_row * x_shape[1], x_shape[3]])
    idx = 0
    for r in range(img_per_row):
        start_row = r * x_shape[1]
        end_row = start_row + x_shape[1]
        output[0:x_shape[2], start_row:end_row] = x_raw[idx]
        idx += 1
        if idx == x_shape[0]:
            break
    idx = 0
    for r in range(img_per_row):
        start_row = r * x_shape[1]
        end_row = start_row + x_shape[1]
        output[x_shape[2]:2 * x_shape[2], start_row:end_row] = x[idx]
        idx += 1
        if idx == x_shape[0]:
            break
    if np.shape(output)[-1] == 1:
        output = np.reshape(output, np.shape(output)[0:2])
    return output


# MAIN

def main():
    # Exp info
    exp_folder = os.path.join(args.output_path, args.dataset, args.exp_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_path = os.path.join(exp_folder, 'checkp')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load train dataset
    x, side_length, channels = load_dataset(args.dataset, args.root_folder)

    inp_shape = [args.batch_size, side_length, side_length, channels]
    print('Number of training samples: {} \nSample shape: {}'.format(x.shape[0], (side_length, side_length, channels)))

    # Model
    model = Resnet(inp_shape, args.num_scale, args.block_per_scale, args.depth_per_block, args.kernel_size,
                   args.base_dim, args.fc_dim, args.latent_dim, args.second_depth, args.second_dim, args.l2_reg)

    writer = tf.summary.create_file_writer(args.output_path)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=None)
    '''
    if not args.val:
        # First stage
        if args.restore:
            # True to restore last checkpoint
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            print('Restored model weights from {}'.format(manager.latest_checkpoint))

            xin = x[:10000]
            _, _, img_recons = model.reconstruct(xin)
            seloss = np.mean(np.square(xin / 255. - img_recons), axis=(1, 2, 3))
            mseloss = np.mean(seloss).astype("float32")
            gamma_x = np.sqrt(mseloss)
            print("mse: ", mseloss)
            mu_z, _ = model.extract_posterior(xin)
            z_hat = model.reconstruct2(mu_z)
            mseloss2 = np.mean(np.square(mu_z - z_hat), axis=(0, 1)).astype("float32")
            gamma_z = np.sqrt(mseloss2)
            print("mse2: ", mseloss2)

        else:
            mseloss = 1.
            gamma_x = 1.
            mseloss2 = 1.
            gamma_z = 1.

        num_sample = np.shape(x)[0]
        iteration_per_epoch = num_sample // args.batch_size

        # First stage: shouldn't backpropagate through second stage variables
        model.encoder2.trainable = False
        model.decoder2.trainable = False

        print('Stage 1 ...')

        for epoch in range(args.epochs):
            np.random.shuffle(x)
            lr = args.lr if args.lr_epochs <= 0 else args.lr * math.pow(args.lr_fac, math.floor(
                float(epoch) / float(args.lr_epochs)))
            epoch_loss = 0

            start_time = time.time()
            for j in range(iteration_per_epoch):
                if j % 100 == 0:
                    print(j)
                image_batch = x[j * args.batch_size:(j + 1) * args.batch_size]
                loss, bmseloss = model.step(1, j, image_batch, gamma_x, lr, ckpt, writer, args.write_iteration)
                epoch_loss += loss
                # print("mse: ", bmseloss)
                # we estimate mse as a weighted combination of the
                # the previous estimation and the minibatch mse'
                mseloss = min(mseloss, mseloss * .99 + bmseloss * .01)
                gamma_x = np.sqrt(mseloss)
                writer.flush()
            end_time = time.time()
            epoch_loss /= iteration_per_epoch
            date = datetime.now()
            print('Date: {date}\t'
                  'Epoch: [Stage 1][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch + 1, args.epochs, epoch_loss, date=date.strftime('%Y-%m-%d %H:%M:%S')))
            print("Gamma_x: ", gamma_x)
            print("mse: ", mseloss)
            print('Time elapsed for current epoch: {}'.format(end_time - start_time))

        # Save first stage weights
        save_path = manager.save()
        print("Saved checkpoint after stage 1 to : {}".format(save_path))

        # Second stage: shouldn't backpropagate through first stage variables
        model.encoder2.trainable = True
        model.decoder2.trainable = True
        model.encoder1.trainable = False
        model.decoder1.trainable = False

        # Second stage
        print('Extracting posterior ...')
        mu_z, sd_z = model.extract_posterior(x)
        idx = np.arange(num_sample)

        print('Stage 2 ...')

        for epoch in range(args.epochs2):
            np.random.shuffle(idx)
            mu_z = mu_z[idx]
            sd_z = sd_z[idx]
            lr = args.lr2 if args.lr_epochs2 <= 0 else args.lr2 * math.pow(args.lr_fac2, math.floor(
                float(epoch) / float(args.lr_epochs2)))
            epoch_loss = 0

            start_time = time.time()
            for j in range(iteration_per_epoch):
                mu_z_batch = mu_z[j * args.batch_size:(j + 1) * args.batch_size]
                sd_z_batch = sd_z[j * args.batch_size:(j + 1) * args.batch_size]
                z_batch = mu_z_batch + sd_z_batch * np.random.normal(0, 1, [args.batch_size, args.latent_dim])
                loss, bmseloss2 = model.step(2, j, z_batch, gamma_z, lr, ckpt, writer, args.write_iteration)
                epoch_loss += loss
                mseloss2 = min(mseloss2, mseloss2 * .99 + bmseloss2 * .01)
                gamma_z = np.sqrt(mseloss2)
                writer.flush()
            end_time = time.time()
            epoch_loss /= iteration_per_epoch
            date = datetime.now()
            print('Date: {date}\t'
                  'Epoch: [Stage 2][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch + 1, args.epochs2, epoch_loss, date=date.strftime('%Y-%m-%d %H:%M:%S')))
            print('Time elapsed for current epoch: {}'.format(end_time - start_time))

        model.encoder1.trainable = True
        model.decoder1.trainable = True

        # Save second stage weights
        save_path = manager.save()
        print("Saved checkpoint after stage 2 to : {}".format(save_path))

    else:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print('Restored model weights from {}'.format(manager.latest_checkpoint))

    # Load test dataset
    x, _, _ = load_test_dataset(args.dataset, args.root_folder)
    print('Number of test samples: {} \nSample shape: {}'.format(x.shape[0], (side_length, side_length, channels)))
    np.random.shuffle(x)
    x = x[:10000]

    print('Reconstructing test images ...')
    zmean, zlogvar, img_recons = model.reconstruct(x)
    x = x.astype('float32') / 255.

    zmeanvar = np.var(zmean, axis=0)
    zlogvarmean = np.mean(np.exp(zlogvar), axis=0)  # check
    zsum = zmeanvar + zlogvarmean

    print('Adjusting ...')
    adjust1 = None
    adjust2 = None
    if args.adjust:
        adjust1 = np.std(x, axis=0)  # None
        z_hat = model.reconstruct2(zmean)
        mse2 = np.mean(np.square(zmean - z_hat))
        var_loss2 = np.mean(zmeanvar) - np.mean(np.var(z_hat, axis=0))
        # We apply an adjustment only if there is an evident variance loss
        if var_loss2 / mse2 > .25:
            # The adjustment is equal to the std provided by var law (close to 1)
            adjust2 = np.sqrt(np.mean(zsum))
        # Fixing reconstructed images
        rescale = np.mean(adjust1) / np.mean(np.std(img_recons, axis=0))
        gmean = np.mean(img_recons, axis=0)
        img_recons = (img_recons - gmean) * rescale + gmean
        # Need to remain in range 0-1
        img_recons = np.maximum(np.minimum(img_recons, 1), 0)
        # Computing adjustments for generated images

    print('Generating ...')
    img_gens1, _ = model.generate(10000, 1, adjust2, adjust1)
    img_gens2, gen_z = model.generate(10000, 2, adjust2, adjust1)

    # if args.fid:
        # print("Rec FID: ", fid.get_fid(x, img_recons.copy()))
        # print("Gen FID (1): ", fid.get_fid(x, img_gens1.copy()))
        # print("Gen FID (2): ", fid.get_fid(x, img_gens2.copy()))

    # img_recons_sample = stich_imgs_2(x, img_recons)
    # img_gens1_sample = stich_imgs(img_gens1)
    # img_gens2_sample = stich_imgs(img_gens2)
    # plt.imsave(os.path.join(exp_folder, 'gen1_sample.jpg'), img_gens1_sample)
    # plt.imsave(os.path.join(exp_folder, 'gen2_sample.jpg'), img_gens2_sample)
    # plt.imsave(os.path.join(exp_folder, 'recon_sample.jpg'), img_recons_sample)

    print("MSE: ", np.mean(np.square(x - img_recons)))
    print("variance law = ", np.mean(zsum))

    count = 0
    for i in range(args.latent_dim):
        # print(zlogvarmean[i])
        if zlogvarmean[i] > 0.8:
            count += 1
    print("Inactive var: ", count)

    for i in range(0, 5):
        imgs = stich_imgs(img_gens2[i * 100:(i + 1) * 100])
        plt.figure(figsize=(20, 20))
        plt.axis("off")
        plt.imshow(imgs)
        # plt.savefig(os.path.join(args.root_folder, "imgs" + str(i) + ".png"), bbox_inches='tight')
        plt.show()
    '''
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print('Restored model weights from {}'.format(manager.latest_checkpoint))

    # DBSRCNN model
    print('************ DBSRCNN model ************')
    # Load train dataset
    # x, _, _ = load_dataset(args.dataset, args.root_folder)

    # Reconstructing celeba train dataset with 2 stage to train DBSRCNN
    print('Reconstructing train dataset for DBSRCNN ...')

    # Saving labels for DBSRCNN model in chunks of 10000
    num_chunk = x.shape[0] // 10000

    # Saving training data for DBSRCNN model in chunks of 10000
    for i in range(num_chunk):
        _, _, recons = model.reconstruct(x[i * 10000:(i + 1) * 10000])

        np.save(os.path.join(args.root_folder, 'data', args.dataset, 'celeba_train_recon' + str(i)), recons)
        print('Saved train data: ', 'celeba_train_recon' + str(i))
        np.save(os.path.join(args.root_folder, 'data', args.dataset, 'celeba_train_labels' + str(i)),
                x[i * 10000:(i + 1) * 10000].astype('float32') / 255.)
        print('Saved label: ', 'celeba_train_labels' + str(i))

    # Reconstructing also test dataset because at line 840 there has been a shuffle so can't use labels in DBSRCNN
    x, _, _ = load_test_dataset(args.dataset, args.root_folder)
    x = x[:10000]

    # Reconstructing celeba test dataset with 2 stage to train DBSRCNN with validation data and labels
    print('Reconstructing test dataset for DBSRCNN ...')
    zmean, zlogvar, test_img_recons = model.reconstruct(x)
    x = x.astype('float32') / 255.

    zmeanvar = np.var(zmean, axis=0)
    zlogvarmean = np.mean(np.exp(zlogvar), axis=0)  # check
    zsum = zmeanvar + zlogvarmean

    print('Adjusting ...')
    adjust1 = None
    adjust2 = None
    if args.adjust:
        adjust1 = np.std(x, axis=0)  # None
        z_hat = model.reconstruct2(zmean)
        mse2 = np.mean(np.square(zmean - z_hat))
        var_loss2 = np.mean(zmeanvar) - np.mean(np.var(z_hat, axis=0))
        # We apply an adjustment only if there is an evident variance loss
        if var_loss2 / mse2 > .25:
            # The adjustment is equal to the std provided by var law (close to 1)
            adjust2 = np.sqrt(np.mean(zsum))
        # Fixing reconstructed images
        rescale = np.mean(adjust1) / np.mean(np.std(test_img_recons, axis=0))
        gmean = np.mean(test_img_recons, axis=0)
        img_recons = (test_img_recons - gmean) * rescale + gmean
        # Need to remain in range 0-1
        test_img_recons = np.maximum(np.minimum(test_img_recons, 1), 0)
        # Computing adjustments for generated images

    np.save(os.path.join(args.root_folder, 'data', args.dataset, 'celeba_test_recon'), test_img_recons)
    print('Saved test data: ', 'celeba_test_recon')

    # Save test dataset to numpy
    np.save(os.path.join(args.root_folder, 'data', args.dataset, 'celeba_test_labels'), x)
    print('Saved test labels: ', 'celeba_test_labels')

    print('Generating ...')
    img_gens1, _ = model.generate(10000, 1, adjust2, adjust1)
    img_gens2, gen_z = model.generate(10000, 2, adjust2, adjust1)

    for i in range(0, 5):
        imgs = stich_imgs(img_gens2[i * 100:(i + 1) * 100])
        plt.figure(figsize=(20, 20))
        plt.axis("off")
        plt.imshow(imgs)
        plt.savefig(os.path.join(args.root_folder, "imgs_gens2-" + str(i) + ".png"), bbox_inches='tight')
        plt.show()

    # Save generated dataset to numpy
    np.save(os.path.join(args.root_folder, 'data', args.dataset, 'celeba_generated_images'), blurred_gen_imgs)
    print('Saved generated data: ', 'celeba_generated_images')

    # Train DBSRCNN
    history = DBSRCNN(input_shape=inp_shape, training=True, generate=False)
    for key in history.history.keys():
        print(key + ': ' + str(history.history[key]))

    # Deblur test dataset
    deblurred_img_recons = DBSRCNN(input_shape=inp_shape, training=False, generate=False)
    deblurred_img_recons = np.maximum(np.minimum(deblurred_img_recons, 1), 0)

    # Deblur generated dataset
    deblurred_img_gen2 = DBSRCNN(input_shape=inp_shape, training=False, generate=True)
    deblurred_img_gen2 = np.maximum(np.minimum(deblurred_img_gen2, 1), 0)

    for i in range(0, 5):
        imgs = stich_imgs(deblurred_img_gen2[i * 100:(i + 1) * 100])
        plt.figure(figsize=(20, 20))
        plt.axis("off")
        plt.imshow(imgs)
        plt.savefig(os.path.join(args.root_folder, "imgs-deblurred" + str(i) + ".png"), bbox_inches='tight')
        plt.show()

    # Results
    print("MSE between test images and images reconstructed by 2StageVAE: ", np.mean(np.square(x - test_img_recons)))
    print("MSE between test images and images reconstructed by 2StageVAE and deblurred: ",
          np.mean(np.square(x - deblurred_img_recons)))
    print('\nPSNRLoss between test images and images reconstructed by 2StageVAE:', PSNRLoss(x, test_img_recons))
    print('PSNRLoss between test images and images reconstructed by 2StageVAE amd deblurred:',
          PSNRLoss(x, deblurred_img_recons))
    print('\nPSNRLoss between test images and images generated by 2StageVAE:', PSNRLoss(x, img_gens2))
    print('PSNRLoss between test images and images generated by 2StageVAE amd deblurred:',
          PSNRLoss(x, deblurred_img_gen2))

    if args.fid:
        print("Rec FID (2StageVAE + DBSRCNN): ", fid.get_fid(x, deblurred_img_recons.copy()))
        print("Gen FID (2) (2StageVAE + DBSRCNN): ", fid.get_fid(x, deblurred_img_gen2.copy()))


##############################################

# dictionary => object (simulate argparse)


#    suggested configurations:
#                 cifar    celeba
#    second_dim   2048     4096
#    num_scale    3        4
#    epochs       700      70
#    lr           .0001    .00005
#    lr_epochs    250      60
#    epochs2      1400     140
#    lr2          .0001    .00005
#    lr_epochs2   400      120
#    l2_reg       None    .001

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


args = {
    "root_folder": '/content/drive/My Drive/VAE/VAE/',
    "output_path": '/content/drive/My Drive/VAE/VAE/experiments',
    "exp_name": 'Exp_1',
    "dataset": 'celeba',
    "gpu": 1,
    "network_structure": 'Resnet',
    "batch_size": 100,
    "write_iteration": 600,
    "latent_dim": 64,
    "second_dim": 4096,
    "second_depth": 3,
    "num_scale": 4,
    "block_per_scale": 1,
    "depth_per_block": 2,
    "kernel_size": 3,
    "base_dim": 32,
    "fc_dim": 512,
    "epochs": 10,
    "lr": .00005,
    "lr_epochs": 60,
    "lr_fac": 0.5,
    "epochs2": 20,
    "lr2": .00005,
    "lr_epochs2": 120,
    "lr_fac2": 0.5,
    "l2_reg": 0.001,
    "val": True,
    "restore": True,
    "adjust": True,
    "fid": True
}
args = Struct(**args)

# preprocess_celeba()

main()