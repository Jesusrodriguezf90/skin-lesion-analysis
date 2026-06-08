"""
dataset.py

Clases de Dataset PyTorch para el proyecto skin-lesion-analysis.

Responsabilidad:
    Proporcionar las clases Dataset reutilizables que encapsulan la carga
    y preprocesamiento de imágenes para ambas tareas del pipeline:
    segmentación (ISIC 2018 Task 1) y clasificación (HAM10000 / Task 3).

    ISICSegmentationDataset — Task 1: imagen + máscara binaria.
    HAM10000Dataset         — Task 3: imagen + etiqueta de clase (7 clases).
    HAM10000MaskedDataset   — Task 3: imagen + imagen enmascarada + etiqueta.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Dataset de segmentación — ISIC 2018 Task 1
# ---------------------------------------------------------------------------


class ISICSegmentationDataset(Dataset):
    """Dataset PyTorch para segmentación binaria de lesiones ISIC 2018 Task 1.

    Carga pares imagen/máscara para entrenar U-Net con encoder ResNet34.
    Cada imagen de Task 1 tiene su máscara ground truth correspondiente
    con el sufijo _segmentation.png anotada manualmente por dermatólogos.

    Args:
        rutas_img: Lista de rutas a las imágenes .jpg de Task 1.
        dir_masks: Directorio que contiene las máscaras _segmentation.png.
        transform: Pipeline de Albumentations a aplicar. Debe incluir
                   ToTensorV2() como último paso.

    Notes:
        Las máscaras se binarizan con umbral 127 — el ground truth de ISIC
        es binario (0/255) pero puede tener artefactos de compresión JPEG.
        La máscara se devuelve con shape (1, H, W) float32, requerido
        por las losses de smp (DiceLoss, SoftBCEWithLogitsLoss).
    """

    def __init__(
        self,
        rutas_img: list,
        dir_masks: Path,
        transform=None,
    ) -> None:
        self.rutas_img = rutas_img
        self.dir_masks = dir_masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rutas_img)

    def __getitem__(self, idx: int) -> tuple:
        """Carga y preprocesa un par imagen/máscara.

        Args:
            idx: Índice del par a cargar.

        Returns:
            Tupla (imagen, máscara) como tensores PyTorch.
            imagen:  (3, H, W) float32 normalizado ImageNet.
            máscara: (1, H, W) float32 binaria {0.0, 1.0}.
        """
        ruta_img  = self.rutas_img[idx]
        ruta_mask = self.dir_masks / (ruta_img.stem + "_segmentation.png")

        img  = np.array(Image.open(ruta_img).convert("RGB"))
        mask = np.array(Image.open(ruta_mask).convert("L"))

        # Binarización de la máscara — float32 requerido por la loss
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # (1, H, W)

        return img, mask


# ---------------------------------------------------------------------------
# Dataset de clasificación — HAM10000 / ISIC 2018 Task 3
# ---------------------------------------------------------------------------


class HAM10000Dataset(Dataset):
    """Dataset PyTorch para clasificación multiclase sobre HAM10000.

    Carga imágenes y etiquetas de las 7 categorías clínicas del challenge
    ISIC 2018 Task 3 para entrenar EfficientNet-B0 con transfer learning.

    Args:
        df:        DataFrame con columnas "ruta" (Path) y "label" (int 0-6).
                   Construir manualmente desde el CSV
                   ISIC2018_Task3_Training_GroundTruth.csv.
        transform: Pipeline de Albumentations a aplicar.

    Notes:
        Las imágenes se cargan en RGB directamente desde disco.
        No se usa caché en memoria — con 10.015 imágenes a 256×256
        ocuparía ~2GB de RAM innecesariamente.
        Las etiquetas son índices enteros correspondientes a:
        0=MEL, 1=NV, 2=BCC, 3=AKIEC, 4=BKL, 5=DF, 6=VASC.
    """

    CLASES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Carga y preprocesa una imagen con su etiqueta.

        Args:
            idx: Índice de la muestra a cargar.

        Returns:
            Tupla (imagen, etiqueta) como tensores PyTorch.
            imagen:   (3, H, W) float32 normalizado ImageNet.
            etiqueta: escalar long (int64).
        """
        ruta  = self.df.loc[idx, "ruta"]
        label = self.df.loc[idx, "label"]

        img = np.array(Image.open(ruta).convert("RGB"))

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset de evaluación — HAM10000 con pseudo-máscaras U-Net
# ---------------------------------------------------------------------------


class HAM10000MaskedDataset(Dataset):
    """Dataset para evaluación comparativa del pipeline completo.

    Devuelve simultáneamente la imagen completa y la imagen enmascarada
    con la pseudo-máscara generada por U-Net, permitiendo evaluar el
    impacto de la segmentación previa sobre la clasificación en un
    único DataLoader vectorizado.

    Usado en 04_evaluation.ipynb para la comparativa:
    Modo A (sin segmentación) vs Modo B (con pseudo-máscara U-Net).

    Args:
        df:        DataFrame con columnas "ruta" (Path) y "label" (int 0-6).
        mascaras:  Array numpy (N, H, W) float32 con pseudo-máscaras
                   binarias {0.0, 1.0} pre-generadas por U-Net.
        transform: Pipeline de Albumentaciones a aplicar a ambas imágenes.
        img_size:  Resolución de las máscaras. Debe coincidir con IMG_SIZE
                   del pipeline (256 en el proyecto).

    Notes:
        El fondo se reemplaza por la media de ImageNet (no por cero)
        para que el modelo normalizado no interprete el fondo como
        señal diagnóstica relevante.
    """

    MEAN_IMG = np.array([0.485, 0.456, 0.406]) * 255

    def __init__(
        self,
        df      : pd.DataFrame,
        mascaras: np.ndarray,
        transform,
        img_size: int,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.mascaras  = mascaras
        self.transform = transform
        self.img_size  = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Carga imagen completa e imagen enmascarada con su etiqueta.

        Args:
            idx: Índice de la muestra a cargar.

        Returns:
            Tupla (t_img, t_masked, etiqueta).
            t_img:    (3, H, W) float32 — imagen completa normalizada.
            t_masked: (3, H, W) float32 — imagen enmascarada normalizada.
            etiqueta: escalar long (int64).
        """
        ruta  = self.df.loc[idx, "ruta"]
        label = self.df.loc[idx, "label"]
        mask  = self.mascaras[idx]

        img_np  = np.array(Image.open(ruta).convert("RGB"))
        img_256 = np.array(
            Image.fromarray(img_np).resize((self.img_size, self.img_size))
        )

        img_masked = img_256.copy().astype(np.float32)
        for c in range(3):
            img_masked[:, :, c] = np.where(
                mask > 0, img_256[:, :, c], self.MEAN_IMG[c]
            )
        img_masked = img_masked.astype(np.uint8)

        t_img    = self.transform(image=img_np)["image"]
        t_masked = self.transform(image=img_masked)["image"]

        return t_img, t_masked, torch.tensor(label, dtype=torch.long)
