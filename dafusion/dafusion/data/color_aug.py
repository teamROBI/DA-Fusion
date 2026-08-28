"""SSD-style photometric augmentation for RGB (train-only, RGB-only).

Ported from the UOAIS baseline (baselines/uoais/adet/data/augmentation.py, itself the
Caffe/SSD `im_transforms` recipe) — the paper-lineage color jitter. Applied directly to the
RGB uint8 array in the DA-Fusion mapper BEFORE the shared geometric transforms, so the depth
modality is never touched (a color op inside the geometric AugmentationList would be replayed
on the metric-XYZ depth via transforms.apply_image and corrupt it).

Reduces overfitting to UOAIS-Sim's rendered appearance -> better sim->real transfer on the
cluttered real benchmarks (OCID especially). Architecture/eval unchanged; train-time only.
"""
import random

import cv2
import numpy as np


class ColorAugSSDTransform:
    """Brightness/contrast/saturation/hue jitter on a uint8 (H,W,3) image.

    Each sub-op fires with prob 0.5; contrast is applied either before or after
    saturation+hue (random order), matching the SSD recipe. Operates internally in BGR/HSV;
    if ``img_format == "RGB"`` the input is swapped to BGR and back so the HSV ops are correct.
    """

    def __init__(self, img_format, brightness_delta=32,
                 contrast_low=0.5, contrast_high=1.5,
                 saturation_low=0.5, saturation_high=1.5, hue_delta=18):
        assert img_format in ("BGR", "RGB")
        self.is_rgb = img_format == "RGB"
        self.brightness_delta = brightness_delta
        self.contrast_low, self.contrast_high = contrast_low, contrast_high
        self.saturation_low, self.saturation_high = saturation_low, saturation_high
        self.hue_delta = hue_delta

    def apply(self, img):
        assert img.ndim == 3 and img.shape[-1] == 3, "ColorAugSSDTransform expects (H,W,3)"
        if self.is_rgb:
            img = img[:, :, ::-1]
        img = self._brightness(img)
        if random.randrange(2):
            img = self._contrast(img)
            img = self._saturation(img)
            img = self._hue(img)
        else:
            img = self._saturation(img)
            img = self._hue(img)
            img = self._contrast(img)
        if self.is_rgb:
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)

    @staticmethod
    def _convert(img, alpha=1.0, beta=0.0):
        img = img.astype(np.float32) * alpha + beta
        return np.clip(img, 0, 255).astype(np.uint8)

    def _brightness(self, img):
        if random.randrange(2):
            return self._convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def _contrast(self, img):
        if random.randrange(2):
            return self._convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def _saturation(self, img):
        if random.randrange(2):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = self._convert(hsv[:, :, 1],
                                         alpha=random.uniform(self.saturation_low, self.saturation_high))
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def _hue(self, img):
        if random.randrange(2):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int)
                            + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img
