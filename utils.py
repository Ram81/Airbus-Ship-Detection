import os
import csv
import random
import logging

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from pipeline_config import EXCLUDED_FILENAMES, ORIGINAL_SIZE


def run_length_encoding(img):
	'''
		img: numpy array 1 - mask, 0 - background
		Returns run length as string formated
	'''
	pixels = img.flatten()
	pixels = np.concatenate([[0], pixels, [0]])
	runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	runs[1::2] -= runs[::2]
	return ' '.join(str(x) for x in runs) 


def run_length_decoding(mask_rle, shape):
	'''
		mask_rle: run-length as string formated (start length)
		shape: (height, width) of array to return
		Returns numpy array 1 - mask, 0 - background
	'''
	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
		img[lo:hi] = 1
	return img.reshape(shape).T


def get_overlayed_mask(image_annotation, shape, labeled=False):
	'''
		image_annotation: RLE masks of image
		shape: shape of masks
		Returns a list of masks from image annotations
	'''
	mask = np.zeros(shape, dtype=np.uint8)
	if image_annotation['EncodedPixels'].any():
		for i, row in image_annotation.reset_index(drop=True).iterrows():
			if labeled:
				label = i + 1
			else:
				label = 1
			mask += label * run_length_decoding(row['EncodedPixels'], shape)
	return mask


def read_masks_from_csv(image_ids, result_file_path, image_sizes):
	'''
		image_ids: list of image ids
		result_file_path: path to test csv file
		image_sizes: list of image shapes
		Returns a list of masks
	'''
	solution = pd.read_csv(result_file_path)
	masks = []
	for imageid, image_size in zip(image_ids, image_sizes):
		image_id_pd = image_id + '.jpg'
		mask = get_overlayed_mask(solution.query('ImageId == @image_id_pd'), image_size, labeled=True)
		masks.append(mask)
	return masks


def overlay_masks(annotation_file_name, target_dir, dev_mode):
	'''
		annotation_file_name: file having annotaion details encoded using RLE
		target_dir: location to store masked annotations
		dev_mode: flag to create a sample set from train data for dev testing

		Creates image masks and stores at target_dir
	'''
	os.makedirs(target_dir, exist_ok=True)
	annotations = pd.read_csv(annotation_file_name)

	if dev_mode:
		annotations = annotations.sample(5, random_state=42)
	for file_name, image_annotation in annotations.groupby("ImageId"):
		if file_name not in EXCLUDED_FILENAMES:
			target_file_name = os.path.join(target_dir, file_name.split('.')[0])
			mask = get_overlayed_mask(image_annotation, ORIGINAL_SIZE)
			plt.imshow(mask)
			plt.show()

			if mask.sum() == 0:
				mask = ORIGINAL_SIZE
			save_target_mask(target_file_name, mask)


def save_target_mask(file_path, mask):
	'''
		using joblib to dump numpy array
	'''
	joblib.dump(mask, file_path)


if __name__ == '__main__':
	overlay_masks('sample.csv', 'masked_image', False)
		
