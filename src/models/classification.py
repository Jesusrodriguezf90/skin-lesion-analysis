"""
classification.py

Constructor del modelo de clasificación EfficientNet-B0.

Responsabilidad:
    Proporcionar la función de construcción del modelo EfficientNet-B0
    preentrenado en ImageNet, encapsulando los parámetros de arquitectura
    usados en 03_classification.ipynb.

    La función build_efficientnet() es el punto de entrada único para
    instanciar el modelo tanto en entrenamiento (pretrained=True) como
    en inferencia (pretrained=False + load_state_dict desde HF Hub).

    Las funciones congelar_encoder() y descongelar_modelo() encapsulan
    la lógica de fine-tuning progresivo — estándar en transfer learning
    para preservar los pesos preentrenados de ImageNet en las primeras
    épocas cuando el clasificador aún genera gradientes erráticos.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import torch
import torch.nn as nn
import timm

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME  = "efficientnet_b0"
DEFAULT_NUM_CLASSES = 7

# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def build_efficientnet(
    model_name : str  = DEFAULT_MODEL_NAME,
    num_classes: int  = DEFAULT_NUM_CLASSES,
    pretrained : bool = True,
    device     : torch.device | None = None,
) -> nn.Module:
    """Construye y devuelve el modelo EfficientNet-B0 para clasificación.

    Encapsula la configuración de arquitectura usada en el proyecto:
    EfficientNet-B0 con transfer learning sobre ImageNet para clasificación
    multiclase de lesiones cutáneas dermoscópicas en 7 categorías clínicas.

    La arquitectura de compound scaling de EfficientNet optimiza anchura,
    profundidad y resolución simultáneamente — alcanza accuracy similar a
    ResNet50 con 5 veces menos parámetros, lo que la hace óptima para
    un dataset de 10.015 imágenes.

    Args:
        model_name:  Nombre del modelo en timm. El proyecto usa
                     "efficientnet_b0" como compromiso entre capacidad
                     y eficiencia para el tamaño del dataset HAM10000.
        num_classes: Número de clases de salida. 7 para las categorías
                     clínicas del challenge ISIC 2018 Task 3:
                     MEL, NV, BCC, AKIEC, BKL, DF, VASC.
        pretrained:  True para cargar pesos preentrenados en ImageNet
                     (entrenamiento). False para inferencia con pesos
                     propios cargados con load_state_dict().
        device:      Dispositivo de cómputo. Si None, usa CUDA si
                     está disponible, CPU en caso contrario.

    Returns:
        Modelo EfficientNet-B0 en el dispositivo especificado.
        El clasificador final tiene num_classes salidas.

    Example:
        >>> # Entrenamiento — pesos ImageNet preentrenados
        >>> model = build_efficientnet(pretrained=True)
        >>> congelar_encoder(model)   # fase 1: solo entrena el head
        >>> model.train()

        >>> # Inferencia — cargar pesos propios desde HF Hub
        >>> model = build_efficientnet(pretrained=False)
        >>> model.load_state_dict(torch.load("best_efficientnet_b0.pth"))
        >>> model.eval()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        model_name,
        pretrained  = pretrained,
        num_classes = num_classes,
    ).to(device)

    return model


def congelar_encoder(model: nn.Module) -> None:
    """Congela todos los parámetros excepto el clasificador final.

    Fase 1 del fine-tuning progresivo: permite entrenar solo el head
    con lr alto sin destruir los pesos preentrenados de ImageNet cuando
    el clasificador aún genera gradientes erráticos al inicio.

    Args:
        model: EfficientNet-B0 cargado con timm. El parámetro
               "classifier" identifica la capa final en timm.

    Notes:
        Con encoder congelado, los parámetros entrenables son ~8.967
        (solo el clasificador lineal) frente a 4.016.515 totales.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def descongelar_modelo(model: nn.Module) -> None:
    """Descongela todos los parámetros del modelo.

    Fase 2 del fine-tuning progresivo: permite ajustar todo el modelo
    con lr reducido (1e-4) para afinar los pesos del encoder sin
    sobreescribir el conocimiento previo de ImageNet.

    Args:
        model: EfficientNet-B0 con encoder congelado previamente
               mediante congelar_encoder().
    """
    for param in model.parameters():
        param.requires_grad = True
