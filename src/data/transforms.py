"""
transforms.py

Pipelines de data augmentation para el proyecto skin-lesion-analysis.

Responsabilidad:
    Proporcionar los pipelines de preprocesamiento y augmentation de imágenes
    usados en entrenamiento y validación de ambos modelos del pipeline:
    U-Net (segmentación) y EfficientNet-B0 (clasificación).

    La normalización usa media/std de ImageNet en ambos casos — tanto
    ResNet34 como EfficientNet-B0 fueron preentrenados con estas estadísticas.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def get_transforms(img_size: int, train: bool) -> A.Compose:
    """Devuelve el pipeline de data augmentation para train o validación.

    Usado por ISICSegmentationDataset y HAM10000Dataset — ambos modelos
    comparten el mismo pipeline de preprocesamiento para garantizar
    consistencia entre entrenamiento e inferencia.

    Args:
        img_size: Resolución de salida (ancho = alto). El proyecto usa
                  256 como compromiso entre detalle y memoria GPU en T4.
        train:    True para data augmentation, False para solo
                  resize + normalización (validación e inferencia).

    Returns:
        Pipeline de Albumentations listo para aplicar sobre imágenes
        numpy (H, W, C) uint8.

    Notes:
        Normalización con media/std de ImageNet — ResNet34 y EfficientNet-B0
        fueron preentrenados con estas estadísticas y esperan el mismo rango.
        Las augmentations geométricas y de color son estándar en CV médico
        para aumentar la variabilidad con datasets pequeños.
        ColorJitter incluido — el color es una feature diagnóstica crítica
        en dermoscopía (marcadores de color asociados a estadios de melanoma).
    """
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
