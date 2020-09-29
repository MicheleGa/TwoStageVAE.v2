import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from scipy.linalg import sqrtm

# Prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3), weights='imagenet')


def get_inception_activations(inps, batch_size=100):
    n_batches = inps.shape[0] // batch_size
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        print('Processing batch: ', i)
        inp = inps[i * batch_size:(i + 1) * batch_size]
        inpr = tf.image.resize(inp, (299, 299))
        act[i * batch_size:(i + 1) * batch_size] = model.predict(inpr, steps=1)
    return act


def get_fid(images1, images2):
    print(images1.shape)
    print(images2.shape)
    print(type(images1))

    # calculate activations
    act1 = get_inception_activations(images1, batch_size=100)

    # print(np.shape(act1))
    act2 = get_inception_activations(images2, batch_size=100)

    # compute mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # compute sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == '__main__':
    model.summary()
