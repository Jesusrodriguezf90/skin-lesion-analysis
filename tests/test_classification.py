"""
Tests unitarios para src/models/classification.py.

Verifica que build_efficientnet() instancia correctamente el modelo,
produce el shape de salida esperado y que congelar_encoder() y
descongelar_modelo() funcionan correctamente. Todos los tests se
ejecutan en CPU sin necesidad de GPU ni del dataset real.
"""

import pytest
import torch

from src.models.classification import (
    build_efficientnet,
    congelar_encoder,
    descongelar_modelo,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMG_SIZE   = 256
BATCH_SIZE = 2
DEVICE     = torch.device("cpu")


@pytest.fixture
def efficientnet_cpu():
    """Instancia EfficientNet-B0 sin pesos preentrenados para tests en CPU."""
    return build_efficientnet(pretrained=False, device=DEVICE)


@pytest.fixture
def batch_sintetico():
    """Batch sintético (B, 3, H, W) float32 para tests de inferencia."""
    return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# Tests de instanciación
# ---------------------------------------------------------------------------


def test_build_efficientnet_instancia():
    """build_efficientnet() debe devolver un nn.Module sin errores."""
    import torch.nn as nn
    model = build_efficientnet(pretrained=False, device=DEVICE)
    assert isinstance(model, nn.Module)


def test_build_efficientnet_model_name_por_defecto():
    """El modelo por defecto debe ser efficientnet_b0."""
    assert DEFAULT_MODEL_NAME == "efficientnet_b0"


def test_build_efficientnet_num_classes_por_defecto():
    """El número de clases por defecto debe ser 7."""
    assert DEFAULT_NUM_CLASSES == 7


def test_build_efficientnet_en_dispositivo_correcto(efficientnet_cpu):
    """El modelo debe estar en el dispositivo especificado (CPU)."""
    for param in efficientnet_cpu.parameters():
        assert param.device.type == "cpu"
        break


def test_build_efficientnet_num_classes_personalizado():
    """build_efficientnet() debe respetar el num_classes especificado."""
    model = build_efficientnet(num_classes=3, pretrained=False, device=DEVICE)
    batch = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    model.eval()
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (1, 3)


# ---------------------------------------------------------------------------
# Tests de forward pass
# ---------------------------------------------------------------------------


def test_efficientnet_output_shape(efficientnet_cpu, batch_sintetico):
    """EfficientNet-B0 debe producir un tensor (B, 7) — logits de 7 clases."""
    efficientnet_cpu.eval()
    with torch.no_grad():
        output = efficientnet_cpu(batch_sintetico)
    assert output.shape == (BATCH_SIZE, DEFAULT_NUM_CLASSES)


def test_efficientnet_output_dtype(efficientnet_cpu, batch_sintetico):
    """La salida de EfficientNet-B0 debe ser float32."""
    efficientnet_cpu.eval()
    with torch.no_grad():
        output = efficientnet_cpu(batch_sintetico)
    assert output.dtype == torch.float32


def test_efficientnet_parametros_totales(efficientnet_cpu):
    """EfficientNet-B0 debe tener aproximadamente 4M parámetros."""
    total = sum(p.numel() for p in efficientnet_cpu.parameters())
    assert 3_000_000 < total < 6_000_000


# ---------------------------------------------------------------------------
# Tests de congelar_encoder / descongelar_modelo
# ---------------------------------------------------------------------------


def test_congelar_encoder_reduce_entrenables(efficientnet_cpu):
    """congelar_encoder() debe reducir significativamente los parámetros entrenables."""
    total_antes = sum(p.numel() for p in efficientnet_cpu.parameters() if p.requires_grad)
    congelar_encoder(efficientnet_cpu)
    total_despues = sum(p.numel() for p in efficientnet_cpu.parameters() if p.requires_grad)
    assert total_despues < total_antes
    # Con encoder congelado quedan ~8.967 parámetros (solo el clasificador)
    assert total_despues < 20_000


def test_descongelar_modelo_restaura_entrenables(efficientnet_cpu):
    """descongelar_modelo() debe restaurar todos los parámetros como entrenables."""
    total_original = sum(p.numel() for p in efficientnet_cpu.parameters())
    congelar_encoder(efficientnet_cpu)
    descongelar_modelo(efficientnet_cpu)
    total_restaurado = sum(p.numel() for p in efficientnet_cpu.parameters() if p.requires_grad)
    assert total_restaurado == total_original


def test_congelar_encoder_solo_classifier_entrenable(efficientnet_cpu):
    """Con encoder congelado solo el clasificador debe ser entrenable."""
    congelar_encoder(efficientnet_cpu)
    for name, param in efficientnet_cpu.named_parameters():
        if param.requires_grad:
            assert "classifier" in name, (
                f"Parámetro {name} es entrenable pero no pertenece al clasificador"
            )


def test_forward_con_encoder_congelado(efficientnet_cpu, batch_sintetico):
    """El modelo debe poder ejecutar un forward pass con encoder congelado."""
    congelar_encoder(efficientnet_cpu)
    efficientnet_cpu.eval()
    with torch.no_grad():
        output = efficientnet_cpu(batch_sintetico)
    assert output.shape == (BATCH_SIZE, DEFAULT_NUM_CLASSES)
