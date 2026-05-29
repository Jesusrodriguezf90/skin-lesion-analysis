"""
Tests de smoke — verificación de importación del stack tecnológico.

Comprueba que todas las dependencias del proyecto están instaladas y son
importables correctamente. Se sustituirá por los tests unitarios reales
de src/ a medida que se implementen los módulos.
"""


def test_torch_importable():
    """torch debe estar instalado y ser importable."""
    import torch
    assert torch.__version__


def test_torchvision_importable():
    """torchvision debe estar instalado y ser importable."""
    import torchvision
    assert torchvision.__version__


def test_segmentation_models_importable():
    """segmentation-models-pytorch debe estar instalado y ser importable."""
    import segmentation_models_pytorch as smp
    assert smp.__version__


def test_timm_importable():
    """timm debe estar instalado y ser importable."""
    import timm
    assert timm.__version__


def test_albumentations_importable():
    """albumentations debe estar instalado y ser importable."""
    import albumentations as A
    assert A.__version__


def test_numpy_importable():
    """numpy debe estar instalado y ser importable."""
    import numpy as np
    assert np.__version__


def test_pillow_importable():
    """Pillow debe estar instalado y ser importable."""
    from PIL import Image
    assert Image.__version__


def test_sklearn_importable():
    """scikit-learn debe estar instalado y ser importable."""
    import sklearn
    assert sklearn.__version__


def test_huggingface_hub_importable():
    """huggingface_hub debe estar instalado y ser importable."""
    import huggingface_hub
    assert huggingface_hub.__version__
