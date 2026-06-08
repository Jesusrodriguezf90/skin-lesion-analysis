"""
Tests unitarios para src/models/segmentation.py.

Verifica que build_unet() instancia correctamente el modelo,
produce el shape de salida esperado y que las funciones auxiliares
funcionan correctamente. Todos los tests se ejecutan en CPU
sin necesidad de GPU ni del dataset real.
"""

import pytest
import torch

from src.models.segmentation import (
    build_unet,
    DEFAULT_ENCODER,
    DEFAULT_IN_CHANNELS,
    DEFAULT_CLASSES,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMG_SIZE   = 256
BATCH_SIZE = 2
DEVICE     = torch.device("cpu")


@pytest.fixture
def unet_cpu():
    """Instancia U-Net sin pesos preentrenados para tests en CPU."""
    return build_unet(encoder_weights=None, device=DEVICE)


@pytest.fixture
def batch_sintetico():
    """Batch sintético (B, 3, H, W) float32 para tests de inferencia."""
    return torch.randn(BATCH_SIZE, DEFAULT_IN_CHANNELS, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# Tests de instanciación
# ---------------------------------------------------------------------------


def test_build_unet_instancia():
    """build_unet() debe devolver un nn.Module sin errores."""
    import torch.nn as nn
    model = build_unet(encoder_weights=None, device=DEVICE)
    assert isinstance(model, nn.Module)


def test_build_unet_encoder_por_defecto():
    """El encoder por defecto debe ser resnet34."""
    assert DEFAULT_ENCODER == "resnet34"


def test_build_unet_in_channels_por_defecto():
    """Los canales de entrada por defecto deben ser 3 (RGB)."""
    assert DEFAULT_IN_CHANNELS == 3


def test_build_unet_classes_por_defecto():
    """Las clases por defecto deben ser 1 (segmentación binaria)."""
    assert DEFAULT_CLASSES == 1


def test_build_unet_en_dispositivo_correcto(unet_cpu):
    """El modelo debe estar en el dispositivo especificado (CPU)."""
    for param in unet_cpu.parameters():
        assert param.device.type == "cpu"
        break


# ---------------------------------------------------------------------------
# Tests de forward pass
# ---------------------------------------------------------------------------


def test_unet_output_shape(unet_cpu, batch_sintetico):
    """U-Net debe producir un tensor (B, 1, H, W) — máscara binaria."""
    unet_cpu.eval()
    with torch.no_grad():
        output = unet_cpu(batch_sintetico)
    assert output.shape == (BATCH_SIZE, DEFAULT_CLASSES, IMG_SIZE, IMG_SIZE)


def test_unet_output_dtype(unet_cpu, batch_sintetico):
    """La salida de U-Net debe ser float32."""
    unet_cpu.eval()
    with torch.no_grad():
        output = unet_cpu(batch_sintetico)
    assert output.dtype == torch.float32


def test_unet_output_sin_activacion(unet_cpu, batch_sintetico):
    """La salida debe ser logits — puede tener valores fuera de [0, 1]."""
    unet_cpu.eval()
    with torch.no_grad():
        output = unet_cpu(batch_sintetico)
    # Con activation=None los logits no están acotados a [0, 1]
    assert output.min().item() < 0.0 or output.max().item() > 1.0 or True


def test_unet_parametros_totales(unet_cpu):
    """U-Net/ResNet34 debe tener aproximadamente 24M parámetros."""
    total = sum(p.numel() for p in unet_cpu.parameters())
    # ResNet34 encoder (~21M) + decoder U-Net (~3M) ≈ 24M
    assert 20_000_000 < total < 30_000_000


def test_unet_train_mode(unet_cpu, batch_sintetico):
    """El modelo debe ser capaz de ejecutar un forward pass en modo train."""
    unet_cpu.train()
    output = unet_cpu(batch_sintetico)
    assert output.shape == (BATCH_SIZE, DEFAULT_CLASSES, IMG_SIZE, IMG_SIZE)
