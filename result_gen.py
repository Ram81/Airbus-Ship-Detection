import os
import csv

from shutil import copyfile

def generate_result_csv(path):
	imageIds = []

	i=0
	with open("train_sample.csv", 'w') as fobj:
		writer = csv.writer(fobj, delimiter=',')
		writer.writerow(["ImageId", "EncodedPixels"])

		with open('train_ship_segmentations_v2.csv', 'r') as input_file:
			reader = csv.reader(input_file, delimiter=',')
			reader.next()
			for row in reader:
				i += 1
				writer.writerow(row)
				imageIds.append(row[0])
				if (i>=20):
					break
		i = 0
		for imageId in imageIds:
			i += 1
			copyfile('train_v2/' + imageId, 'train/' + imageId)
			if (i>=20):
				break
		print(len(imageIds))


if __name__ == '__main__':
	generate_result_csv('train_v2/')