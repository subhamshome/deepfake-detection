import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

# Define GLCM params
distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['contrast', 'dissimilarity',
              'homogeneity', 'energy', 'correlation']

# Define LBP params
radius = 3
n_points = 10 * radius
method = 'uniform'

# Define Gabor params
kernel_sizes = [5, 11, 15, 21]
thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
sigmas = [1, 3, 5]
lambdas = [5, 10, 15]
gammas = [0.05, 0.5, 0.95]
psis = [0]


# Gabor feature vector
def extract_gabor_features(img, kernel_sizes, thetas, sigmas, lambdas, gammas, psis):

    feature_vector = []

    for kernel_size in kernel_sizes:
        for theta in thetas:
            for sigma in sigmas:
                for lambd in lambdas:
                    for gamma in gammas:
                        for psi in psis:
                            kernel = cv2.getGaborKernel(
                                (kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                            filtered_img = cv2.filter2D(
                                img, cv2.CV_8UC3, kernel)

                            mean = np.mean(filtered_img)
                            std_dev = np.std(filtered_img)

                            feature_vector.append(mean)
                            feature_vector.append(std_dev)

    feature_vector = np.array(feature_vector)

    return feature_vector


# DFT feature vector
def extract_dft_features(img):

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mag_spectrum = 20 * \
        np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    band1 = mag_spectrum[100:150, 100:150]
    band2 = mag_spectrum[150:200, 150:200]
    band3 = mag_spectrum[200:250, 200:250]

    mean1 = np.mean(band1)
    mean2 = np.mean(band2)
    mean3 = np.mean(band3)

    var1 = np.var(band1)
    var2 = np.var(band2)
    var3 = np.var(band3)

    skew1 = np.nan_to_num(np.mean((band1 - mean1)**3) /
                          (np.mean((band1 - mean1)**2)**(3/2)))
    skew2 = np.nan_to_num(np.mean((band2 - mean2)**3) /
                          (np.mean((band2 - mean2)**2)**(3/2)))
    skew3 = np.nan_to_num(np.mean((band3 - mean3)**3) /
                          (np.mean((band3 - mean3)**2)**(3/2)))

    kurt1 = np.nan_to_num(np.mean((band1 - mean1)**4) /
                          (np.mean((band1 - mean1)**2)**2))
    kurt2 = np.nan_to_num(np.mean((band2 - mean2)**4) /
                          (np.mean((band2 - mean2)**2)**2))
    kurt3 = np.nan_to_num(np.mean((band3 - mean3)**4) /
                          (np.mean((band3 - mean3)**2)**2))

    feature_vector = np.array(
        [mean1, mean2, mean3, var1, var2, var3, skew1, skew2, skew3, kurt1, kurt2, kurt3])

    return feature_vector


# GLCM feature vector
def extract_glcm_features(img):
    glcm = graycomatrix(img, distances, angles, levels=256,
                        symmetric=True, normed=True)
    glcm_features = np.hstack(
        [graycoprops(glcm, prop).ravel() for prop in properties])

    return glcm_features


# LBP Feature vector
def extract_lbp_features(img):
    lbp = local_binary_pattern(img, n_points, radius, method)
    lbp_features, _ = np.histogram(lbp.ravel(), bins=np.arange(
        0, n_points + 3), range=(0, n_points + 2))

    return lbp_features


# Main function
def featureExtractor(folder):
    feat_vects = []
    for filename in os.listdir(folder):
        if filename.endswith('.bmp') or filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename),
                             cv2.IMREAD_GRAYSCALE)

            glcm_feats = extract_glcm_features(img)
            lbp_feats = extract_lbp_features(img)
            gabor_feats = extract_gabor_features(
                img, kernel_sizes, thetas, sigmas, lambdas, gammas, psis)
            dft_feats = extract_dft_features(img)

            real_feature_vector = np.concatenate(
                [glcm_feats, lbp_feats, gabor_feats, dft_feats])

            feat_vects.append(real_feature_vector)

    return feat_vects
