"""
segmentation.py

Constructor del modelo de segmentación U-Net/ResNet34.

Responsabilidad:
    Proporcionar la función de construcción del modelo U-Net con encoder
    ResNet34 preentrenado en ImageNet, encapsulando los parámetros de
    arquitectura usados en 02_segmentation.ipynb.

    La función build_unet() es el punto de entrada único para instanciar
    el modelo tanto en entrenamiento (encoder_weights="imagenet") como
    en inferencia (encoder_weights=None + load_state_dict desde HF Hub).

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------

DEFAULT_ENCODER         = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"
DEFAULT_IN_CHANNELS     = 3
DEFAULT_CLASSES         = 1

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def build_unet(
    encoder_name   : str        = DEFAULT_ENCODER,
    encoder_weights: str | None = DEFAULT_ENCODER_WEIGHTS,
    in_channels    : int        = DEFAULT_IN_CHANNELS,
    classes        : int        = DEFAULT_CLASSES,
    device         : torch.device | None = None,
) -> nn.Module:
    """Construye y devuelve el modelo U-Net con encoder ResNet34.

    Encapsula la configuración de arquitectura usada en el proyecto:
    U-Net con encoder ResNet34 preentrenado en ImageNet para segmentación
    binaria de lesiones cutáneas dermoscópicas.

    La arquitectura encoder-decoder de U-Net con skip connections permite
    recuperar el detalle espacial de los bordes de lesión que se pierde
    en la compresión del encoder — crítico para la segmentación precisa
    de bordes irregulares en dermoscopía.

    Args:
        encoder_name:    Nombre del encoder backbone. ResNet34 es el
                         compromiso óptimo entre capacidad y velocidad
                         para el tamaño de dataset ISIC 2018 Task 1.
        encoder_weights: Pesos de preentrenamiento del encoder. Usar
                         "imagenet" para entrenamiento (transfer learning)
                         o None para inferencia con pesos propios cargados
                         posteriormente con load_state_dict().
        in_channels:     Número de canales de entrada. 3 para imágenes RGB.
        classes:         Número de clases de salida. 1 para segmentación
                         binaria (lesión vs fondo).
        device:          Dispositivo de cómputo. Si None, usa CUDA si
                         está disponible, CPU en caso contrario.

    Returns:
        Modelo U-Net en el dispositivo especificado.
        El parámetro activation=None indica que la loss aplica sigmoid
        internamente (SoftBCEWithLogitsLoss, DiceLoss de smp).

    Example:
        >>> # Entrenamiento — encoder preentrenado en ImageNet
        >>> model = build_unet(encoder_weights="imagenet")
        >>> model.train()

        >>> # Inferencia — cargar pesos propios desde HF Hub
        >>> model = build_unet(encoder_weights=None)
        >>> model.load_state_dict(torch.load("best_unet_resnet34.pth"))
        >>> model.eval()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name    = encoder_name,
        encoder_weights = encoder_weights,
        in_channels     = in_channels,
        classes         = classes,
        activation      = None,   # sigmoid se aplica en la loss
    ).to(device)

    return model
