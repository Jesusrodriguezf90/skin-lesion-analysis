"""
app.py

Demo Gradio del pipeline de análisis de lesiones cutáneas dermoscópicas.

Responsabilidad:
    Exponer el pipeline U-Net → EfficientNet-B0 como interfaz interactiva.
    El usuario puede subir una imagen dermoscópica propia o seleccionar
    uno de los 7 ejemplos del dataset ISIC 2018 (uno por clase clínica).

    El pipeline opera en dos modos comparativos:
    - Modo A: clasificación sobre imagen completa (sin segmentación)
    - Modo B: clasificación sobre región segmentada por U-Net (pipeline real)

    Los modelos se descargan automáticamente desde HF Hub al arrancar.
    No se requiere token de usuario — ambos repositorios son públicos.

    Advertencia clínica:
    Este sistema está diseñado exclusivamente para imágenes dermoscópicas
    capturadas con dermatoscopio estándar. No es aplicable a fotografías
    clínicas convencionales ni fotos de smartphone sin dermatoscopio.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp
import timm
from huggingface_hub import hf_hub_download
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TITULO = "Skin Lesion Analysis — Pipeline Dermoscópico"

DESCRIPCION = """
**Pipeline de análisis de lesiones cutáneas en dos etapas** sobre ISIC 2018:

1. **Segmentación** — U-Net/ResNet34 delimita la región de la lesión (Dice 0.8982)
2. **Clasificación** — EfficientNet-B0 clasifica en 7 categorías clínicas (AUC 0.9719)

La demo muestra ambos modos en paralelo para ilustrar el impacto de la segmentación previa:
- **Modo A:** clasificación sobre imagen completa
- **Modo B:** clasificación sobre región segmentada por U-Net

> ⚠️ **Requisito de imagen:** solo imágenes dermoscópicas capturadas con dermatoscopio estándar.
> No válido para fotografías clínicas convencionales ni fotos de smartphone sin dermatoscopio.
> Resolución mínima recomendada: 256×256px.

> ⚕️ **Aviso:** esta demo es exclusivamente educativa y de portfolio. No constituye diagnóstico médico.
"""

CLASES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NOMBRES_CLASES = {
    "MEL"  : "Melanoma",
    "NV"   : "Melanocytic nevus",
    "BCC"  : "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis",
    "BKL"  : "Benign keratosis",
    "DF"   : "Dermatofibroma",
    "VASC" : "Vascular lesion",
}

IMG_SIZE     = 256
DEVICE       = torch.device("cpu")  # HF Spaces CPU Basic
HF_REPO_UNET = "Jesusrodriguezf90/unet-resnet34-isic2018-segmentation"
HF_REPO_EFF  = "Jesusrodriguezf90/efficientnet-b0-isic2018-classification"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
MEAN_IMG      = IMAGENET_MEAN * 255

# Ejemplos del dataset ISIC 2018 — uno por clase clínica (mediana de cada clase)
EJEMPLOS = [
    ["examples/ISIC_0030824.jpg"],  # MEL
    ["examples/ISIC_0029186.jpg"],  # NV
    ["examples/ISIC_0029123.jpg"],  # BCC
    ["examples/ISIC_0028730.jpg"],  # AKIEC
    ["examples/ISIC_0029217.jpg"],  # BKL
    ["examples/ISIC_0029760.jpg"],  # DF
    ["examples/ISIC_0029742.jpg"],  # VASC
]

# ---------------------------------------------------------------------------
# Carga de modelos — se ejecuta una única vez al arrancar el Space
# ---------------------------------------------------------------------------

def cargar_modelos() -> tuple:
    """Descarga y carga ambos modelos desde HF Hub.

    Los modelos son públicos — no requieren token de usuario.
    La descarga ocurre una sola vez al arrancar el Space y queda
    en caché en el sistema de ficheros de HF Spaces.

    Returns:
        Tupla (unet, efficientnet) listos para inferencia en CPU.
    """
    print("Descargando modelos desde HF Hub...")

    ruta_unet = hf_hub_download(
        repo_id  = HF_REPO_UNET,
        filename = "best_unet_resnet34.pth",
    )
    ruta_effnet = hf_hub_download(
        repo_id  = HF_REPO_EFF,
        filename = "best_efficientnet_b0.pth",
    )

    # U-Net con encoder ResNet34 — misma arquitectura que en 02_segmentation.ipynb
    unet = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(DEVICE)
    unet.load_state_dict(torch.load(ruta_unet, map_location=DEVICE))
    unet.eval()

    # EfficientNet-B0 — misma arquitectura que en 03_classification.ipynb
    effnet = timm.create_model(
        "efficientnet_b0",
        pretrained  = False,
        num_classes = 7,
    ).to(DEVICE)
    effnet.load_state_dict(torch.load(ruta_effnet, map_location=DEVICE))
    effnet.eval()

    print("Modelos cargados correctamente")
    return unet, effnet


# Carga global al arrancar — evita recargar en cada inferencia
UNET, EFFNET = cargar_modelos()

# Pipeline de preprocesamiento — idéntico al usado en entrenamiento
TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ToTensorV2(),
])

# ---------------------------------------------------------------------------
# Funciones de inferencia
# ---------------------------------------------------------------------------


def preprocesar_imagen(img_pil: Image.Image) -> tuple:
    """Preprocesa una imagen PIL para inferencia.

    Convierte la imagen a numpy, aplica el transform de validación
    (resize + normalización ImageNet) y devuelve el tensor listo
    para pasar por los modelos.

    Args:
        img_pil: Imagen PIL en formato RGB.

    Returns:
        Tupla (img_np, tensor) donde img_np es el array original
        y tensor es el tensor (1, 3, H, W) normalizado.
    """
    img_np = np.array(img_pil.convert("RGB"))
    tensor = TRANSFORM(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    return img_np, tensor


def generar_mascara(tensor: torch.Tensor) -> np.ndarray:
    """Genera la pseudo-máscara binaria con U-Net.

    Args:
        tensor: Tensor de imagen (1, 3, H, W) normalizado.

    Returns:
        Array numpy (H, W) float32 binario {0.0, 1.0}.
    """
    with torch.no_grad():
        pred = torch.sigmoid(UNET(tensor))
    return (pred > 0.5).float().squeeze().cpu().numpy()


def aplicar_mascara(img_np: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Aplica la máscara sobre la imagen — fondo a media ImageNet.

    El fondo se reemplaza por la media de ImageNet en lugar de cero
    para que el modelo normalizado no interprete el fondo como señal
    diagnóstica relevante.

    Args:
        img_np: Array numpy (H, W, 3) uint8 original.
        mask:   Array numpy (H, W) float32 binario {0.0, 1.0}.

    Returns:
        Tensor (1, 3, H, W) float32 normalizado con fondo neutro.
    """
    img_256 = np.array(
        Image.fromarray(img_np).resize((IMG_SIZE, IMG_SIZE))
    )
    img_masked = img_256.copy().astype(np.float32)
    for c in range(3):
        img_masked[:, :, c] = np.where(
            mask > 0, img_256[:, :, c], MEAN_IMG[c]
        )
    return TRANSFORM(image=img_masked.astype(np.uint8))["image"].unsqueeze(0).to(DEVICE)


def clasificar(tensor: torch.Tensor) -> tuple:
    """Clasifica un tensor de imagen con EfficientNet-B0.

    Args:
        tensor: Tensor (1, 3, H, W) normalizado.

    Returns:
        Tupla (prediccion, probabilidades) donde prediccion es el
        índice de la clase con mayor probabilidad y probabilidades
        es un dict {nombre_clase: probabilidad} para gr.Label.
    """
    with torch.no_grad():
        logits = EFFNET(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred  = probs.argmax()
    label = {
        f"{c} — {NOMBRES_CLASES[c]}": float(probs[i])
        for i, c in enumerate(CLASES)
    }
    return pred, label


def visualizar_mascara(
    img_np: np.ndarray,
    mask  : np.ndarray,
) -> np.ndarray:
    """Genera una visualización de la máscara superpuesta sobre la imagen.

    Dibuja el contorno de la lesión en verde sobre la imagen original
    para facilitar la interpretación visual de la segmentación.

    Args:
        img_np: Array numpy (H, W, 3) uint8 original.
        mask:   Array numpy (H, W) float32 binario {0.0, 1.0}.

    Returns:
        Array numpy (H, W, 3) uint8 con contorno de segmentación.
    """
    import cv2

    img_resize = np.array(
        Image.fromarray(img_np).resize((IMG_SIZE, IMG_SIZE))
    ).copy()

    mask_uint8  = (mask * 255).astype(np.uint8)
    contornos, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_resize, contornos, -1, (0, 255, 0), 2)

    return img_resize


# ---------------------------------------------------------------------------
# Función principal de predicción — llamada por Gradio
# ---------------------------------------------------------------------------


def predecir(imagen_input) -> tuple:
    """Ejecuta el pipeline completo sobre una imagen de entrada.

    Orquesta los dos modos de clasificación (A y B) y devuelve
    todos los outputs necesarios para la interfaz Gradio.

    Args:
        imagen_input: Imagen de entrada — puede ser ruta (str) cuando
                      el usuario selecciona un ejemplo, o array numpy
                      cuando el usuario sube una imagen propia.

    Returns:
        Tupla de 5 elementos para los outputs de Gradio:
        - imagen_mascara:   visualización de la segmentación U-Net
        - label_modo_a:     dict de probabilidades modo A (sin segm.)
        - resultado_modo_a: string con predicción modo A
        - label_modo_b:     dict de probabilidades modo B (con segm.)
        - resultado_modo_b: string con predicción modo B
    """
    # Aceptar tanto ruta de archivo como array numpy
    if isinstance(imagen_input, str):
        img_pil = Image.open(imagen_input).convert("RGB")
    else:
        img_pil = Image.fromarray(imagen_input).convert("RGB")

    # Preprocesar
    img_np, tensor = preprocesar_imagen(img_pil)

    # Generar pseudo-máscara con U-Net
    mask = generar_mascara(tensor)

    # Modo A — imagen completa → EfficientNet
    pred_a, probs_a = clasificar(tensor)
    clase_a = CLASES[pred_a]
    texto_a = f"**{clase_a}** — {NOMBRES_CLASES[clase_a]}"

    # Modo B — imagen enmascarada → EfficientNet
    tensor_masked   = aplicar_mascara(img_np, mask)
    pred_b, probs_b = clasificar(tensor_masked)
    clase_b = CLASES[pred_b]
    texto_b = f"**{clase_b}** — {NOMBRES_CLASES[clase_b]}"

    # Visualización de la máscara
    img_mascara = visualizar_mascara(img_np, mask)

    return img_mascara, probs_a, texto_a, probs_b, texto_b

# ---------------------------------------------------------------------------
# Interfaz Gradio
# ---------------------------------------------------------------------------

NOTA_PIPELINE = """
### Sobre los resultados

Los dos modos permiten comparar el impacto de la segmentación previa:

- **Modo A** clasifica sobre la imagen completa — EfficientNet usa tanto la lesión como el contexto de piel circundante
- **Modo B** clasifica sobre la región segmentada por U-Net — el fondo se reemplaza por la media de ImageNet

La evaluación experimental sobre 10.015 imágenes mostró que la segmentación previa degrada la clasificación (ΔF1 -0.486).
La causa principal es que EfficientNet aprendió a usar el contexto de piel sana como información diagnóstica complementaria,
que se pierde al enmascarar el fondo. Resultado consistente con Hasan et al. (2020).

> ⚕️ **Aviso:** esta demo es exclusivamente educativa y de portfolio. No constituye diagnóstico médico.
"""

with gr.Blocks(title=TITULO) as demo:

    gr.Markdown(f"# {TITULO}")
    gr.Markdown(DESCRIPCION)

    with gr.Row():

        # ── Columna izquierda — entrada ──────────────────────────────────
        with gr.Column(scale=1):
            imagen_input = gr.Image(
                label       = "Imagen dermoscópica",
                type        = "numpy",
                sources     = ["upload"],
                height      = 300,
            )
            analizar_btn = gr.Button("Analizar", variant="primary")

            gr.Examples(
                examples   = EJEMPLOS,
                inputs     = imagen_input,
                label      = "Ejemplos del dataset ISIC 2018 — un caso por clase clínica",
                examples_per_page = 7,
            )

        # ── Columna central — segmentación ───────────────────────────────
        with gr.Column(scale=1):
            imagen_mascara = gr.Image(
                label  = "Segmentación U-Net (contorno en verde)",
                type   = "numpy",
                height = 300,
            )
            gr.Markdown(
                "U-Net/ResNet34 preentrenado en ISIC 2018 Task 1 "
                "(2.594 imágenes). **Dice 0.8982 · IoU 0.8293**"
            )

        # ── Columna derecha — clasificación ──────────────────────────────
        with gr.Column(scale=2):

            with gr.Row():

                with gr.Column():
                    gr.Markdown("### Modo A — Sin segmentación")
                    resultado_modo_a = gr.Markdown(
                        value = "*Pendiente de análisis*"
                    )
                    label_modo_a = gr.Label(
                        label     = "Probabilidades por clase",
                        num_top_classes = 7,
                    )

                with gr.Column():
                    gr.Markdown("### Modo B — Con pseudo-máscara U-Net")
                    resultado_modo_b = gr.Markdown(
                        value = "*Pendiente de análisis*"
                    )
                    label_modo_b = gr.Label(
                        label     = "Probabilidades por clase",
                        num_top_classes = 7,
                    )

    gr.Markdown(NOTA_PIPELINE)

    # ── Eventos ──────────────────────────────────────────────────────────

    analizar_btn.click(
        fn      = predecir,
        inputs  = [imagen_input],
        outputs = [
            imagen_mascara,
            label_modo_a,
            resultado_modo_a,
            label_modo_b,
            resultado_modo_b,
        ],
    )

    # Analizar también al seleccionar un ejemplo
    imagen_input.change(
        fn      = predecir,
        inputs  = [imagen_input],
        outputs = [
            imagen_mascara,
            label_modo_a,
            resultado_modo_a,
            label_modo_b,
            resultado_modo_b,
        ],
    )


if __name__ == "__main__":
    demo.launch()
