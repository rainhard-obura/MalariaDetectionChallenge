import kornia.augmentation as K
from kornia.constants import Resample


def train_augment_and_transform(image_size, config):
    """
    Creates a composition of Kornia augmentations for object detection tasks based on config.

    Args:
    cfg (dict): Configuration dictionary containing augmentation parameters.

    Returns:
    K.AugmentationSequential: A sequential composition of Kornia augmentations.
    """
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=config["hflip_prob"]),
        K.RandomVerticalFlip(p=config["vflip_prob"]),
        K.RandomRotation(
            degrees=config["rotate_degrees"],
            p=config["rotate_prob"],
            resample=Resample.BILINEAR,
            align_corners=True,
        ),
        K.RandomResizedCrop(
            size=image_size,
            scale=(config["crop_scale_min"], config["crop_scale_max"]),
            p=config["crop_prob"],
        ),
        K.RandomGaussianBlur(
            kernel_size=config["blur_kernel_size"],
            sigma=config["blur_sigma"],
            p=config["blur_prob"],
        ),
        K.ColorJitter(
            brightness=config["brightness"],
            contrast=config["contrast"],
            saturation=config["saturation"],
            hue=config["hue"],
            p=config["color_jitter_prob"],
        ),
        K.RandomGrayscale(p=config["gray_prob"]),
        K.RandomPlasmaBrightness(
            roughness=config["plasma_brightness_roughness"],
            p=config["plasma_brightness_prob"],
        ),
        K.RandomPosterize(bits=config["posterize_bits"], p=config["posterize_prob"]),
        K.RandomSolarize(
            thresholds=config["solarize_threshold"],
            additions=config["solarize_addition"],
            p=config["solarize_prob"],
        ),
        K.RandomAffine(
            degrees=config["affine_degrees"],
            translate=config["affine_translate"],
            scale=config["affine_scale"],
            shear=config["affine_shear"],
            p=config["affine_prob"],
        ),
        K.RandomHue(hue=config["hue_range"], p=config["hue_prob"]),
        data_keys=["input", "bbox"],
        same_on_batch=False,
    )


def validation_transform(cfg):
    return None
