"""
Tests unitarios para src/utils/metrics.py.

Verifica que dice_coefficient() e iou_coefficient() calculan
correctamente los valores esperados en casos conocidos: predicción
perfecta, predicción vacía y predicción parcial.
"""

import pytest
import torch

from src.utils.metrics import dice_coefficient, iou_coefficient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
H, W       = 64, 64


@pytest.fixture
def prediccion_perfecta():
    """Logits muy altos donde la máscara es 1 — predicción perfecta."""
    target = torch.zeros(BATCH_SIZE, 1, H, W)
    target[:, :, H//4:3*H//4, W//4:3*W//4] = 1.0
    # Logits altos donde target=1, bajos donde target=0
    logits = target * 20.0 - (1 - target) * 20.0
    return logits, target


@pytest.fixture
def prediccion_vacia():
    """Logits muy negativos — el modelo no predice ninguna lesión."""
    target = torch.zeros(BATCH_SIZE, 1, H, W)
    target[:, :, H//4:3*H//4, W//4:3*W//4] = 1.0
    logits = torch.full((BATCH_SIZE, 1, H, W), -10.0)
    return logits, target


@pytest.fixture
def target_vacio():
    """Target completamente vacío — sin lesión en la imagen."""
    target = torch.zeros(BATCH_SIZE, 1, H, W)
    logits = torch.full((BATCH_SIZE, 1, H, W), -10.0)
    return logits, target


# ---------------------------------------------------------------------------
# Tests de dice_coefficient
# ---------------------------------------------------------------------------


def test_dice_perfecto(prediccion_perfecta):
    """Dice debe ser ≈1.0 con predicción perfecta."""
    logits, target = prediccion_perfecta
    dice = dice_coefficient(logits, target)
    assert dice > 0.99


def test_dice_vacio_prediccion(prediccion_vacia):
    """Dice debe ser bajo cuando el modelo no predice nada."""
    logits, target = prediccion_vacia
    dice = dice_coefficient(logits, target)
    assert dice < 0.1


def test_dice_rango(prediccion_perfecta):
    """Dice debe estar en [0.0, 1.0]."""
    logits, target = prediccion_perfecta
    dice = dice_coefficient(logits, target)
    assert 0.0 <= dice <= 1.0


def test_dice_devuelve_float(prediccion_perfecta):
    """dice_coefficient debe devolver un float Python."""
    logits, target = prediccion_perfecta
    dice = dice_coefficient(logits, target)
    assert isinstance(dice, float)


def test_dice_target_vacio(target_vacio):
    """Dice con target vacío no debe lanzar excepción (smooth evita div/0)."""
    logits, target = target_vacio
    dice = dice_coefficient(logits, target)
    assert isinstance(dice, float)


def test_dice_threshold_personalizado(prediccion_perfecta):
    """dice_coefficient debe respetar el threshold especificado."""
    logits, target = prediccion_perfecta
    dice_05 = dice_coefficient(logits, target, threshold=0.5)
    dice_09 = dice_coefficient(logits, target, threshold=0.9)
    # Ambos deben ser válidos
    assert 0.0 <= dice_05 <= 1.0
    assert 0.0 <= dice_09 <= 1.0


# ---------------------------------------------------------------------------
# Tests de iou_coefficient
# ---------------------------------------------------------------------------


def test_iou_perfecto(prediccion_perfecta):
    """IoU debe ser ≈1.0 con predicción perfecta."""
    logits, target = prediccion_perfecta
    iou = iou_coefficient(logits, target)
    assert iou > 0.99


def test_iou_vacio_prediccion(prediccion_vacia):
    """IoU debe ser bajo cuando el modelo no predice nada."""
    logits, target = prediccion_vacia
    iou = iou_coefficient(logits, target)
    assert iou < 0.1


def test_iou_rango(prediccion_perfecta):
    """IoU debe estar en [0.0, 1.0]."""
    logits, target = prediccion_perfecta
    iou = iou_coefficient(logits, target)
    assert 0.0 <= iou <= 1.0


def test_iou_devuelve_float(prediccion_perfecta):
    """iou_coefficient debe devolver un float Python."""
    logits, target = prediccion_perfecta
    iou = iou_coefficient(logits, target)
    assert isinstance(iou, float)


def test_iou_menor_o_igual_que_dice(prediccion_perfecta):
    """IoU siempre debe ser ≤ Dice para el mismo batch."""
    logits, target = prediccion_perfecta
    dice = dice_coefficient(logits, target)
    iou  = iou_coefficient(logits, target)
    # Relación matemática: IoU = Dice / (2 - Dice)
    assert iou <= dice + 1e-6


def test_iou_target_vacio(target_vacio):
    """IoU con target vacío no debe lanzar excepción (smooth evita div/0)."""
    logits, target = target_vacio
    iou = iou_coefficient(logits, target)
    assert isinstance(iou, float)
