"""
metrics.py

Métricas de evaluación para segmentación de lesiones cutáneas.

Responsabilidad:
    Proporcionar las funciones de métricas usadas durante el entrenamiento
    y evaluación de U-Net: coeficiente Dice e índice IoU (Jaccard).

    Ambas métricas se calculan sobre batches completos y devuelven
    la media del batch como float, para uso directo en los loops
    de entrenamiento de 02_segmentation.ipynb.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import torch

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Calcula el coeficiente Dice entre predicción y ground truth.

    Mide el solapamiento entre la máscara predicha y la máscara ground
    truth. Es la métrica principal de evaluación en segmentación médica
    porque penaliza tanto los falsos positivos como los falsos negativos
    de forma equilibrada.

    Args:
        pred:      Tensor de predicciones con logits (B, 1, H, W).
                   Se aplica sigmoid internamente antes de binarizar.
        target:    Tensor de máscaras ground truth (B, 1, H, W) float32
                   con valores binarios {0.0, 1.0}.
        threshold: Umbral para binarizar la predicción tras sigmoid.
                   El valor por defecto 0.5 es el estándar en ISIC 2018.

    Returns:
        Dice medio del batch como float en [0.0, 1.0].
        1.0 indica solapamiento perfecto, 0.0 indica sin solapamiento.

    Notes:
        Smooth=1 evita división por cero en máscaras vacías — estándar
        en implementaciones de Dice para segmentación médica.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    smooth   = 1.0
    intersec = (pred_bin * target).sum(dim=(2, 3))
    dice     = (2.0 * intersec + smooth) / (
        pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )
    return dice.mean().item()


def iou_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Calcula el índice IoU (Jaccard) entre predicción y ground truth.

    Métrica complementaria al Dice — más estricta porque penaliza más
    los píxeles fuera de la intersección. Es la métrica de ranking
    oficial del challenge ISIC 2018 Task 1.

    Args:
        pred:      Tensor de predicciones con logits (B, 1, H, W).
                   Se aplica sigmoid internamente antes de binarizar.
        target:    Tensor de máscaras ground truth (B, 1, H, W) float32
                   con valores binarios {0.0, 1.0}.
        threshold: Umbral para binarizar la predicción tras sigmoid.

    Returns:
        IoU medio del batch como float en [0.0, 1.0].
        1.0 indica solapamiento perfecto, 0.0 indica sin solapamiento.

    Notes:
        Smooth=1 evita división por cero en máscaras vacías.
        Relación con Dice: IoU = Dice / (2 - Dice), por lo que
        IoU siempre es menor o igual que Dice para el mismo batch.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    smooth   = 1.0
    intersec = (pred_bin * target).sum(dim=(2, 3))
    union    = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersec
    iou      = (intersec + smooth) / (union + smooth)
    return iou.mean().item()
