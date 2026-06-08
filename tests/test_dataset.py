"""
Tests unitarios para src/data/dataset.py y src/data/transforms.py.

Verifica que las clases Dataset instancian correctamente, devuelven
los shapes esperados y que los transforms producen tensores válidos.
Todos los tests son ejecutables sin GPU y sin el dataset real —
se usan imágenes sintéticas generadas en memoria.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from PIL import Image

from src.data.transforms import get_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.data.dataset import (
    ISICSegmentationDataset,
    HAM10000Dataset,
    HAM10000MaskedDataset,
)

# ---------------------------------------------------------------------------
# Fixtures — datos sintéticos en memoria
# ---------------------------------------------------------------------------

IMG_SIZE = 256


@pytest.fixture
def imagen_sintetica_path(tmp_path):
    """Crea una imagen RGB sintética en disco y devuelve su ruta."""
    img = Image.fromarray(
        np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
    )
    ruta = tmp_path / "ISIC_0000001.jpg"
    img.save(ruta)
    return ruta


@pytest.fixture
def mascara_sintetica_path(tmp_path):
    """Crea una máscara binaria sintética en disco y devuelve su ruta."""
    mask = Image.fromarray(
        np.random.choice([0, 255], size=(450, 600)).astype(np.uint8)
    )
    ruta = tmp_path / "ISIC_0000001_segmentation.png"
    mask.save(ruta)
    return ruta


@pytest.fixture
def df_sintetico(tmp_path):
    """Crea un DataFrame con una fila sintética y su imagen en disco."""
    img = Image.fromarray(
        np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
    )
    ruta = tmp_path / "ISIC_0024306.jpg"
    img.save(ruta)
    return pd.DataFrame({"ruta": [ruta], "label": [0]})


# ---------------------------------------------------------------------------
# Tests de transforms
# ---------------------------------------------------------------------------


def test_get_transforms_train_devuelve_compose():
    """get_transforms(train=True) debe devolver un pipeline de Albumentations."""
    import albumentations as A
    t = get_transforms(IMG_SIZE, train=True)
    assert isinstance(t, A.Compose)


def test_get_transforms_val_devuelve_compose():
    """get_transforms(train=False) debe devolver un pipeline de Albumentations."""
    import albumentations as A
    t = get_transforms(IMG_SIZE, train=False)
    assert isinstance(t, A.Compose)


def test_get_transforms_produce_tensor_correcto():
    """El pipeline de transforms debe producir un tensor (3, H, W) float32."""
    transform = get_transforms(IMG_SIZE, train=False)
    img = np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
    resultado = transform(image=img)["image"]
    assert isinstance(resultado, torch.Tensor)
    assert resultado.shape == (3, IMG_SIZE, IMG_SIZE)
    assert resultado.dtype == torch.float32


def test_imagenet_mean_std_correctos():
    """Las constantes de normalización deben ser las de ImageNet."""
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD  == (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Tests de ISICSegmentationDataset
# ---------------------------------------------------------------------------


def test_segmentation_dataset_len(imagen_sintetica_path, mascara_sintetica_path):
    """ISICSegmentationDataset debe devolver la longitud correcta."""
    ds = ISICSegmentationDataset(
        rutas_img = [imagen_sintetica_path],
        dir_masks = mascara_sintetica_path.parent,
    )
    assert len(ds) == 1


def test_segmentation_dataset_shapes(imagen_sintetica_path, mascara_sintetica_path):
    """ISICSegmentationDataset debe devolver tensores con shapes correctos."""
    transform = get_transforms(IMG_SIZE, train=False)
    ds = ISICSegmentationDataset(
        rutas_img = [imagen_sintetica_path],
        dir_masks = mascara_sintetica_path.parent,
        transform = transform,
    )
    img, mask = ds[0]
    assert img.shape  == (3, IMG_SIZE, IMG_SIZE)
    assert mask.shape == (1, IMG_SIZE, IMG_SIZE)


def test_segmentation_dataset_mask_binaria(imagen_sintetica_path, mascara_sintetica_path):
    """La máscara debe ser binaria con valores {0.0, 1.0}."""
    transform = get_transforms(IMG_SIZE, train=False)
    ds = ISICSegmentationDataset(
        rutas_img = [imagen_sintetica_path],
        dir_masks = mascara_sintetica_path.parent,
        transform = transform,
    )
    _, mask = ds[0]
    valores_unicos = torch.unique(mask).tolist()
    assert all(v in [0.0, 1.0] for v in valores_unicos)


def test_segmentation_dataset_tipos(imagen_sintetica_path, mascara_sintetica_path):
    """La imagen debe ser float32 y la máscara float32."""
    transform = get_transforms(IMG_SIZE, train=False)
    ds = ISICSegmentationDataset(
        rutas_img = [imagen_sintetica_path],
        dir_masks = mascara_sintetica_path.parent,
        transform = transform,
    )
    img, mask = ds[0]
    assert img.dtype  == torch.float32
    assert mask.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests de HAM10000Dataset
# ---------------------------------------------------------------------------


def test_ham10000_dataset_len(df_sintetico):
    """HAM10000Dataset debe devolver la longitud correcta."""
    ds = HAM10000Dataset(df_sintetico)
    assert len(ds) == 1


def test_ham10000_dataset_shapes(df_sintetico):
    """HAM10000Dataset debe devolver imagen (3,H,W) y etiqueta escalar."""
    transform = get_transforms(IMG_SIZE, train=False)
    ds = HAM10000Dataset(df_sintetico, transform=transform)
    img, label = ds[0]
    assert img.shape == (3, IMG_SIZE, IMG_SIZE)
    assert label.shape == torch.Size([])


def test_ham10000_dataset_etiqueta_tipo(df_sintetico):
    """La etiqueta debe ser un tensor long (int64)."""
    transform = get_transforms(IMG_SIZE, train=False)
    ds = HAM10000Dataset(df_sintetico, transform=transform)
    _, label = ds[0]
    assert label.dtype == torch.long


def test_ham10000_dataset_clases():
    """HAM10000Dataset.CLASES debe contener las 7 categorías de ISIC 2018."""
    assert HAM10000Dataset.CLASES == ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


# ---------------------------------------------------------------------------
# Tests de HAM10000MaskedDataset
# ---------------------------------------------------------------------------


def test_masked_dataset_shapes(df_sintetico):
    """HAM10000MaskedDataset debe devolver (img, img_masked, label) con shapes correctos."""
    transform = get_transforms(IMG_SIZE, train=False)
    mascaras = np.random.randint(0, 2, (1, IMG_SIZE, IMG_SIZE)).astype(np.float32)
    ds = HAM10000MaskedDataset(df_sintetico, mascaras, transform, IMG_SIZE)
    t_img, t_masked, label = ds[0]
    assert t_img.shape    == (3, IMG_SIZE, IMG_SIZE)
    assert t_masked.shape == (3, IMG_SIZE, IMG_SIZE)
    assert label.dtype    == torch.long


def test_masked_dataset_len(df_sintetico):
    """HAM10000MaskedDataset debe devolver la longitud correcta."""
    transform = get_transforms(IMG_SIZE, train=False)
    mascaras  = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ds = HAM10000MaskedDataset(df_sintetico, mascaras, transform, IMG_SIZE)
    assert len(ds) == 1
