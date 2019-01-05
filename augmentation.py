import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

from .utils import get_crop_pad_sequence


def _perspective_transform_augment_images(self, images, random_state, parents, hooks):
    result = images
    if not self.keep_size:
        result = list(result)

    matrices, max_heights, max_widths = self._create_matrices(
        [image.shape for image in images],
        random_state
    )

    for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
        warped = cv2.warpPerspective(images[i], M, (max_width, max_height))
        if warped.ndim == 2 and images[i].ndim == 3:
            warped = np.expand_dims(warped, 2)
        if self.keep_size:
            h, w = images[i].shape[0:2]
            warped = ia.imresize_single_image(warped, (h, w))

        result[i] = warped

    return result


iaa.PerspectiveTransform._augment_images = _perspective_transform_augment_images

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-20, 20),
                           translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='symmetric'),
                ]),
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.02, 0.04))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.10))),
], random_order=True)

intensity_seq = iaa.Sequential([
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-30, 30)),
                iaa.AddElementwise((-30, 30)),
                iaa.Multiply((0.9, 1.1)),
                iaa.MultiplyElementwise((0.9, 1.1)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 2.0)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.MedianBlur(k=(3, 7))
        ])
    ])
], random_order=False)


def crop_seq(crop_size):
    seq = iaa.Sequential([affine_seq,
                          RandomCropFixedSize(px=crop_size)], random_order=False)
    return seq


def padding_seq(pad_size, pad_method):
    seq = iaa.Sequential([PadFixed(pad=pad_size, pad_method=pad_method),
                          ]).to_deterministic()
    return seq


def pad_to_fit_net(divisor, pad_mode, rest_of_augs=iaa.Noop()):
    return iaa.Sequential(InferencePad(divisor, pad_mode), rest_of_augs)


class PadFixed(iaa.Augmenter):
    PAD_FUNCTION = {'reflect': cv2.BORDER_REFLECT_101,
                    'replicate': cv2.BORDER_REPLICATE,
                    }

    def __init__(self, pad=None, pad_method=None, name=None, deterministic=False, random_state=None):
        super().__init__(name, deterministic, random_state)
        self.pad = pad
        self.pad_method = pad_method

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        for i, image in enumerate(images):
            image_pad = self._pad(image)
            result.append(image_pad)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _pad(self, img):
        img_ = img.copy()

        if self._is_expanded_grey_format(img):
            img_ = np.squeeze(img_, axis=-1)

        h_pad, w_pad = self.pad
        img_ = cv2.copyMakeBorder(img_.copy(), h_pad, h_pad, w_pad, w_pad, PadFixed.PAD_FUNCTION[self.pad_method])

        if self._is_expanded_grey_format(img):
            img_ = np.expand_dims(img_, axis=-1)

        return img_

    def get_parameters(self):
return []