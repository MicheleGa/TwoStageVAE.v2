import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input, Conv2D
import os
import numpy as np
import math


def log10(x):
    numerator = np.log(x)
    denominator = np.log(k.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNRLoss(y_true, y_pred):
    """
        PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
        It can be calculated as
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
        When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
        However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
        Thus we remove that component completely and only compute the remaining MSE component.

        """

    return 10.0 * k.log(1.0 / k.mean(k.square(y_pred - y_true))) / k.log(10.0)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def DBSRCNN(input_shape, training=None, generate=None):
    print('**** DBSRCNN model ****')
    input_shape = input_shape[1:]
    root_folder = '/content/drive/My Drive/VAE/VAE/'
    batch_size = 100
    model_path = os.path.join(root_folder, 'DBSRCNN_data/')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    hdf5_folder = os.path.join(model_path, 'weights/')

    if not os.path.exists(hdf5_folder):
        os.makedirs(hdf5_folder)

    ckpt_folder = os.path.join(model_path, 'ckpt/')

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    # DBSRCNN model
    x = Input(shape=input_shape)
    c1 = Conv2D(filters=32, kernel_size=(9, 9), padding='same', kernel_initializer='he_normal', activation='relu')(x)
    c2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu')(c1)
    m = tf.keras.layers.concatenate([c1, c2])
    c3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu')(m)
    c4 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu')(c3)
    c5 = Conv2D(filters=3, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal')(c4)

    model = tf.keras.Model(inputs=x, outputs=c5)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[PSNRLoss])

    if training:

        datas = [f for f in os.listdir(os.path.join(root_folder, 'data', 'celeba')) if
                 f.startswith('celeba_train_recon')]
        labels = [f for f in os.listdir(os.path.join(root_folder, 'data', 'celeba')) if
                  f.startswith('celeba_train_labels')]

        for data in datas:
            num_data = data[18] + data[19]
            if data[19] == '.':
                num_data = data[18]
            for label in labels:
                num_label = label[19] + label[20]
                if label[20] == '.':
                    num_label = label[19]

                if num_data == num_label:
                    print('Processing ' + data + ' with ' + label)
                    # Train images
                    blurred_train_imgs = np.load(os.path.join(root_folder, 'data', 'celeba', data))
                    train_images_labels = np.load(os.path.join(root_folder, 'data', 'celeba', label))

                    train_dataset = (
                        tf.data.Dataset.from_tensor_slices((blurred_train_imgs, train_images_labels)).batch(
                            batch_size=batch_size))

                    # Test images
                    blurred_test_imgs = np.load(os.path.join(root_folder, 'data', 'celeba', 'celeba_test_recon.npy'))
                    test_images_labels = np.load(os.path.join(root_folder, 'data', 'celeba', 'celeba_test_labels.npy'))

                    test_dataset = (
                        tf.data.Dataset.from_tensor_slices((blurred_test_imgs, test_images_labels)).batch(
                            batch_size=batch_size))

                    print('Number of training samples: {}'.format(blurred_train_imgs.shape[0]))
                    print('Input shape: ', input_shape)
                    print('Number of test samples: {}'.format(blurred_test_imgs.shape[0]))
                    print('Input shape: ', input_shape)

                    # Model check point callback
                    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_folder,
                                                                             save_weights_only=True,
                                                                             verbose=1)

                    # Early stopping callback
                    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                               patience=3,
                                                                               mode='min',
                                                                               verbose=1)

                    # Learning schedule callback
                    lrate_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

                    # Fit model
                    history = model.fit(train_dataset,
                                        callbacks=[checkpoint_callback,
                                                   early_stopping_callback,
                                                   lrate_callback],
                                        verbose=1,
                                        epochs=10000,
                                        shuffle=True,
                                        validation_data=test_dataset)

        # Save model and weights
        model.save_weights(os.path.join(model_path, 'weights', 'DBSRCNNmodel_wegihts_blur1'))

        # Return parameters of training
        return history

    else:
        # Load model weights
        model.load_weights(ckpt_folder).expect_partial()

        if not generate:
            # Test images
            blurred_test_imgs = np.load(os.path.join(root_folder, 'data', 'celeba', 'celeba_test_recon.npy'))
            test_images_labels = np.load(os.path.join(root_folder, 'data', 'celeba', 'celeba_test_labels.npy'))

            test_dataset = (
                tf.data.Dataset.from_tensor_slices((blurred_test_imgs, test_images_labels)).batch(
                    batch_size=batch_size))

            loss, acc = model.evaluate(test_dataset, verbose=1)
            print('DBSRCNN model loss: {} and PSNRLoss: {}'.format(loss, acc))

            print('Number of test samples: {}'.format(blurred_test_imgs.shape[0]))
            print('Input shape: ', input_shape)
            # Return deblurred test dataset

            return model.predict(blurred_test_imgs, verbose=1)

        else:
            # Generated images
            blurred_generated_imgs = np.load(os.path.join(root_folder, 'data', 'celeba', 'celeba_generated_images.npy'))

            print('Number of generated images: {}'.format(blurred_generated_imgs.shape[0]))
            print('Input shape: ', input_shape)

            # Return deblurred generated images
            return model.predict(blurred_generated_imgs, verbose=1)
