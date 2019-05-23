import numpy as np
from PIL import ImageEnhance
from PIL import Image
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage import transform
from skimage import io
import tensorflow as tf
import matplotlib
import os
from skimage.measure import compare_ssim as ssim
import argparse
from random import uniform
import copy

def rotate(img, rad_angle):
    afine_tf = transform.AffineTransform(rotation=rad_angle)
    rotated_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return rotated_img

def translate(img, trans_x, trans_y):
    afine_tf = transform.AffineTransform(translation=(trans_x, trans_y))
    translated_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return translated_img

def scale(img, scale_1, scale_2):
    afine_tf = transform.AffineTransform(scale=(scale_1, scale_2))
    scaled_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return scaled_img

def shear(img, value):
    afine_tf = transform.AffineTransform(shear=value)
    sheared_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return sheared_img
    
def blur(img, sigma):
    is_colour = len(img.shape)==3
    blur_img = np.uint8(rescale_intensity(gaussian(img, sigma=sigma, multichannel=is_colour,preserve_range=True),out_range=(0,255)))
    return blur_img

def change_brightness(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Brightness(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_color(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Color(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_contrast(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Contrast(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_sharpness(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Sharpness(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def apply_mutation(img, delta, mutation_prob):
    '''
    apply pixel perturbation to an image
    :param delta: the pixel changes could be a random value sampled from [-delta,delta]
    :param mutation_prob: the probability of applying a perturbation to one pixel
    :return: a mutated image
    '''
    normalized_img = normalize(img)
    shape = img.shape
    U = np.random.uniform(size=shape)*2*delta - delta
    mask = np.random.binomial(1, mutation_prob, size=shape)
    mutation = mask * U
    mutated_img = normalized_img + mutation
    denormlized_img = denormlize(mutated_img)
    return denormlized_img

def build_transformation_metadata(): 
    '''
    construct a dictionary containing the interval boundaries for valid transformations' parameters
    :return: the dictionary of transformations' metadata
    '''
    tr_metadata = {
        'sigma_min_bound':0.0,
        'sigma_max_bound':0.5,
        'contrast_min_bound':1.0,
        'contrast_max_bound':2.0,
        'brightness_min_bound':1.0,
        'brightness_max_bound':2.0,
        'sharpness_min_bound':1.0,
        'sharpness_max_bound':2.0,
        'delta':0.05,
        'mutation_prob': 0.01,
        'scale_max_bound': 1.0,
        'scale_min_bound': 0.95,
        'trans_abs_bound': 4, # absolute bound means ==> min: -4, max: 4
        'shear_abs_bound':0.1,
        'rot_angle_abs_bound': np.pi/45
    }
    return tr_metadata

# to complete
def apply_random_transformation(image_origin, tr_metadata):
    '''
    apply a sequence of image transformations to the image 
    :return transformed_data: a set of transformed images
    :return ssim_value: the ssim value comparing between the original image and the transformed one w.r.t pixel-value transformations
    '''
    transformed_data = []
    transformed_image = image_origin.copy()
    # apply mutation using the metadata parameters 
    transformed_image = apply_mutation(transformed_image, tr_metadata['delta'], tr_metadata['mutation_prob'])
    # change contrast using random factor within the valid interval
    contrast_factor = uniform(tr_metadata['contrast_min_bound'], tr_metadata['contrast_max_bound'])
    transformed_image = change_contrast(transformed_image, contrast_factor)
    # change brightness using random factor within the valid interval
    brightness_factor = uniform(tr_metadata['brightness_min_bound'], tr_metadata['brightness_max_bound'])
    transformed_image = change_brightness(transformed_image, brightness_factor)
    # change sharpness using random factor within the valid interval
    sharpness_factor = uniform(tr_metadata['sharpness_min_bound'], tr_metadata['sharpness_max_bound'])
    transformed_image = change_sharpness(transformed_image, sharpness_factor)
    # add blur effect using random sigma within the valid interval
    sigma = uniform(tr_metadata['sigma_min_bound'], tr_metadata['sigma_max_bound'])
    transformed_image = blur(transformed_image, sigma)
    # compute the ssim between the original image and the resulting pixel-value transformed image
    ssim_value = ssim(image_origin, transformed_image)
    # add the transformed image to the transformed data
    transformed_data.append(transformed_image)
    # translate the image using random translation parameters within the valid interval
    trans_abs_bound = tr_metadata['trans_abs_bound']
    trans_x = uniform(-trans_abs_bound, trans_abs_bound)
    trans_y = uniform(-trans_abs_bound, trans_abs_bound)
    translated_image = translate(image_origin.copy(), trans_x, trans_y)
    # add the translated image to the transformed data
    transformed_data.append(translated_image)
    # scale the image using random scale parameters within the valid interval
    scale_min_bound = tr_metadata['scale_min_bound']
    scale_max_bound = tr_metadata['scale_max_bound']
    scale1 = uniform(scale_min_bound, scale_max_bound)
    scale2 = uniform(scale_min_bound, scale_max_bound)
    scaled_image = scale(image_origin.copy(), scale1, scale2)
    # add the scaled image to the transformed data
    transformed_data.append(scaled_image)
    # rotate the image using random angle  within the valid interval
    rot_angle_abs_bound = tr_metadata['rot_angle_abs_bound']
    rad_angle = uniform(-rot_angle_abs_bound, rot_angle_abs_bound)
    rotated_image = rotate(image_origin.copy(), rad_angle)
    # add the rotated image to the transformed data
    transformed_data.append(rotated_image)
    # shear the image using random value within the valid interval
    shear_abs_bound = tr_metadata['shear_abs_bound']
    shear_value = uniform(-shear_abs_bound, shear_abs_bound)
    sheared_image = shear(image_origin.copy(), shear_value)
    # add the sheared image to the transformed data
    transformed_data.append(sheared_image)
    return transformed_data, ssim_value

def normalize(img):
    norm_img = np.float32(img / 255.0)
    return norm_img

def denormlize(img):
    denorm_img = np.uint8(img * 255.0)
    return denorm_img

def store_data(id, data):
    if not os.path.isdir('./test_images'):
        os.mkdir('./test_images')
    matplotlib.image.imsave("./test_images/id_{}.png".format(id), data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', help='help')
    parser.add_argument('--attempts', help='help')
    args = vars(parser.parse_args())
    ssim_threshold = float(args['threshold'])
    attempts_count = int(args['attempts'])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    tr_meta = build_transformation_metadata()
    for i in range(attempts_count):
        image_idx = np.random.choice(len(x_test))
        image = x_test[image_idx]
        mutated_images, ssim_value = apply_random_transformation(image, tr_meta)
        if ssim_value > ssim_threshold:
            store_data(i, image)
      