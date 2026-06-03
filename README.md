# Skin Lesion Analysis — Segmentación y Clasificación con Deep Learning

**Autor:** Jesús Rodríguez  
**Dataset:** ISIC 2018 — Skin Lesion Analysis Towards Melanoma Detection  
**Stack:** PyTorch · U-Net/ResNet34 · EfficientNet-B0 · Gradio · GitHub Actions

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Funcionalidades](#funcionalidades)
- [Arquitectura](#arquitectura)
- [Stack Tecnológico](#stack-tecnológico)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Puesta en Marcha](#instalación-y-puesta-en-marcha)
- [Resultados](#resultados)
- [Datos de prueba](#datos-de-prueba)
- [Hoja de Ruta](#hoja-de-ruta)
- [Limitaciones](#limitaciones)
- [Licencia](#licencia)

---

## Descripción

**Skin Lesion Analysis** es un pipeline de análisis de lesiones cutáneas dermoscópicas en dos etapas implementado con PyTorch:

1. **Segmentación** — U-Net con encoder ResNet34 preentrenado en ImageNet para delimitar píxel a píxel la región de la lesión en la imagen dermoscópica
2. **Clasificación** — EfficientNet-B0 entrenado sobre la región segmentada para clasificar la lesión en 7 categorías clínicas

Cada modelo se entrena con su dataset específico del challenge ISIC 2018. En inferencia, el pipeline opera en secuencia: U-Net segmenta la lesión → EfficientNet clasifica sobre la región segmentada.

---

## Funcionalidades

> Las funcionalidades marcadas con 🔲 están especificadas y pendientes de implementación.

- ✅ Exploración y análisis del dataset ISIC 2018 (notebook 01_eda)
- 🔲 Segmentación de lesiones con U-Net/ResNet34 y métricas Dice e IoU (notebook 02_segmentation)
- 🔲 Clasificación multi-clase con EfficientNet-B0 sobre región segmentada (notebook 03_classification)
- 🔲 Evaluación comparativa con métricas AUC, F1 y análisis de errores (notebook 04_evaluation)
- 🔲 Publicación de modelos en HF Hub
- 🔲 Demo Gradio en HF Spaces

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│              Imagen dermoscópica (224×224)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              U-Net (encoder ResNet34)                       │
│         Preentrenado en ImageNet — fine-tuning ISIC 2018    │
│         Entrenado con Task 1 (2.594 imágenes + máscaras)    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Máscara binaria de la lesión                   │
│              Segmentación píxel a píxel                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              EfficientNet-B0                                │
│         Clasificación de la región segmentada               │
│         Entrenado con Task 3 / HAM10000 (10.015 imágenes)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Clasificación (7 categorías clínicas)               │
│  Melanoma · Nevus · BCC · AK · BK · Dermatofibroma · Vasc. │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack Tecnológico

| Componente | Herramienta | Propósito |
|---|---|---|
| Framework DL | [PyTorch](https://pytorch.org/) | Entrenamiento y inferencia de modelos |
| Segmentación | [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) | U-Net con encoder ResNet34 preentrenado |
| Clasificación | [timm](https://github.com/huggingface/pytorch-image-models) | EfficientNet-B0 con transfer learning |
| Augmentación | [Albumentations](https://albumentations.ai/) | Augmentación de imágenes médicas |
| Entrenamiento | [Kaggle Notebooks](https://www.kaggle.com/code) | GPU T4 gratuita para entrenamiento |
| Demo | [Gradio](https://gradio.app/) + [HF Spaces](https://huggingface.co/spaces) | Interfaz interactiva pública |
| CI/CD | [GitHub Actions](https://github.com/features/actions) | Tests automáticos en cada push |

---

## Estructura del Proyecto

```
skin-lesion-analysis/
│
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline CI/CD — tests automáticos en cada push
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploración del dataset ISIC 2018
│   ├── 02_segmentation.ipynb    # U-Net con ResNet34 encoder
│   ├── 03_classification.ipynb  # EfficientNet sobre región segmentada
│   └── 04_evaluation.ipynb      # Métricas Dice, IoU, AUC, F1
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch Dataset para ISIC 2018
│   │   └── transforms.py        # Augmentación de imágenes médicas
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segmentation.py      # U-Net con ResNet34 encoder
│   │   └── classification.py    # EfficientNet-B0 clasificador
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # Dice, IoU, AUC, F1
│       └── visualization.py    # Visualización de máscaras y predicciones
│
├── data/
│   ├── raw/
│   │   ├── ISIC2018_Task1-2_Training_Input/      # 2.594 imágenes .jpg (excluidas de Git)
│   │   ├── ISIC2018_Task1_Training_GroundTruth/  # 2.594 máscaras .png (excluidas de Git)
│   │   └── ISIC2018_Task3_Training_GroundTruth.csv  # Etiquetas HAM10000 (10.015 entradas)
│   └── processed/               # Artefactos generados por los notebooks (excluidos de Git)
│
├── app/
│   └── app.py                   # Demo Gradio en HF Spaces
│
├── tests/
│   ├── test_imports.py          # Tests de smoke — verificación del stack
│   ├── test_dataset.py          # Tests unitarios de ISICDataset
│   ├── test_segmentation.py     # Tests unitarios del modelo de segmentación
│   └── test_classification.py  # Tests unitarios del modelo de clasificación
│
├── .env.example                 # Plantilla de variables de entorno
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Instalación y Puesta en Marcha

### Requisitos previos

- Python 3.10+
- Cuenta en [Kaggle](https://www.kaggle.com/) para descarga del dataset y entrenamiento con GPU
- Cuenta en [Hugging Face](https://huggingface.co/join) con token de API generado

### Variables de entorno

Copia `.env.example` a `.env` y añade tu token de Hugging Face:

```env
HF_TOKEN=tu_token_de_huggingface_aquí
```

> El archivo `.env` está excluido de Git por defecto. No lo subas al repositorio bajo ninguna circunstancia.

### Ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Jesusrodriguezf90/skin-lesion-analysis.git
cd skin-lesion-analysis

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Descarga del dataset

El pipeline utiliza dos datasets independientes del challenge ISIC 2018, cada uno para una tarea distinta:

- **Task 1** (imágenes + máscaras de segmentación → entrena U-Net): descarga desde [challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)
- **Task 3 / HAM10000** (10.015 imágenes + etiquetas → entrena EfficientNet): descarga desde [challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)

> **Por qué dos datasets distintos:** Task 1 y Task 3 son conjuntos disjuntos — no comparten imágenes. Task 1 proporciona 2.594 imágenes con máscaras de segmentación píxel a píxel anotadas manualmente. Task 3 / HAM10000 proporciona 10.015 imágenes con etiquetas de clasificación clínica. Cada modelo se entrena con el dataset que contiene las anotaciones que necesita. En inferencia el pipeline opera en secuencia sobre cualquier imagen nueva.

Una vez descargados, coloca los archivos en `data/raw/` siguiendo la estructura indicada en la sección anterior.

### Entrenamiento en Kaggle

1. Sube los datasets a Kaggle como dataset privado
2. Ejecuta los notebooks en orden: `01_eda` → `02_segmentation` → `03_classification` → `04_evaluation`

---

## Resultados

| Tarea | Métrica | Valor |
|---|---|---|
| Segmentación | Dice | — |
| Segmentación | IoU (Jaccard) | — |
| Clasificación | AUC | — |
| Clasificación | F1 (macro) | — |

*Resultados pendientes de entrenamiento completo.*

---

## Datos de prueba

> El dataset utilizado para el desarrollo y evaluación del pipeline es el benchmark de referencia mundial para análisis automatizado de lesiones cutáneas dermoscópicas.

**ISIC 2018 — Skin Lesion Analysis Towards Melanoma Detection**  
International Skin Imaging Collaboration (ISIC)  
🔗 https://challenge.isic-archive.com/data/#2018

| Dataset | Imágenes | Máscaras | Etiquetas | Uso |
|---|---|---|---|---|
| Task 1 (entrenamiento) | 2.594 | ✅ | ❌ | Entrena U-Net |
| Task 3 / HAM10000 | 10.015 | ❌ | ✅ | Entrena EfficientNet |
| Task 1 (validación) | 100 | ❌ | ❌ | — |
| Task 1 (test) | 1.000 | ❌ | ❌ | — |

**Categorías (Task 3):** Melanoma · Melanocytic nevus · Basal cell carcinoma · Actinic keratosis · Benign keratosis · Dermatofibroma · Vascular lesion

**Condiciones de uso:**

El dataset se utiliza exclusivamente con fines de investigación y desarrollo bajo los términos de uso de ISIC. Las imágenes no se distribuyen ni se incluyen en el repositorio — están excluidas del control de versiones mediante `.gitignore`.

---

## Hoja de Ruta

| Estado | Componente |
|---|---|
| ✅ | Estructura del proyecto y documentación |
| ✅ | Exploración del dataset (notebook 01_eda) |
| 🔲 | Segmentación U-Net (notebook 02_segmentation) |
| 🔲 | Clasificación EfficientNet (notebook 03_classification) |
| 🔲 | Evaluación comparativa (notebook 04_evaluation) |
| 🔲 | Publicación de modelos en HF Hub |
| 🔲 | Demo Gradio en HF Spaces |

---

## Limitaciones

- El modelo está entrenado con imágenes **dermoscópicas estándar** capturadas con dermatoscopio — no es aplicable a fotografías clínicas convencionales ni fotos de smartphone sin dermatoscopio
- La escala física de la lesión en milímetros no se usa como feature — el modelo aprende patrones de textura, color y forma relativos a la imagen
- El dataset ISIC 2018 está fuertemente desbalanceado — NV (nevus) representa el 66.9% de Task 3, mientras que DF (dermatofibroma) es solo el 1.1%
- Task 1 y Task 3 son conjuntos disjuntos — U-Net y EfficientNet se entrenan con datos independientes sin solapamiento

---

## Referencias

- Codella et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC). *arXiv:1902.03368*
- Tschandl et al. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*

---

## Licencia

Este proyecto está distribuido bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  Construido para la medicina y la inteligencia artificial
</p>
