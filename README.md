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
- ✅ Segmentación de lesiones con U-Net/ResNet34 y métricas Dice e IoU (notebook 02_segmentation)
- ✅ Clasificación multi-clase con EfficientNet-B0 con fine-tuning progresivo (notebook 03_classification)
- ✅ Evaluación del pipeline completo con análisis crítico (notebook 04_evaluation)
- ✅ Publicación de modelos en HF Hub
- 🔲 Demo Gradio en HF Spaces

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│              Imagen dermoscópica (256×256)                   │
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

> **Resolución de entrada 256×256:** compromiso entre detalle preservado y memoria GPU disponible en T4. Las imágenes originales de ISIC 2018 tienen una resolución media de ~3089×2111px — reducir a 224×224 supone un factor de compresión de ~14x que pierde detalle en bordes finos de lesión. 512×512 obliga a batch size ≤4, alargando el entrenamiento sin mejora proporcional con 2.594 imágenes. 256×256 es el compromiso estándar en la literatura para este dataset con encoders preentrenados.

---

## Stack Tecnológico

| Componente | Herramienta | Propósito |
|---|---|---|
| Framework DL | [PyTorch](https://pytorch.org/) | Entrenamiento e inferencia de modelos |
| Segmentación | [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) | U-Net con encoder ResNet34 preentrenado |
| Clasificación | [timm](https://github.com/huggingface/pytorch-image-models) | EfficientNet-B0 con transfer learning |
| Data augmentation | [Albumentations](https://albumentations.ai/) | Data augmentation de imágenes médicas |
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
│   ├── 03_classification.ipynb  # EfficientNet-B0 con fine-tuning progresivo
│   └── 04_evaluation.ipynb      # Evaluación del pipeline completo
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch Dataset para ISIC 2018
│   │   └── transforms.py        # Data augmentation de imágenes médicas
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
2. Añade `HF_TOKEN` en Kaggle → Add-ons → Secrets
3. Ejecuta los notebooks en orden: `01_eda` → `02_segmentation` → `03_classification` → `04_evaluation`

---

## Resultados

| Tarea | Métrica | Valor |
|---|---|---|
| Segmentación | Dice | **0.8982** |
| Segmentación | IoU (Jaccard) | **0.8293** |
| Clasificación | F1 macro | **0.7700** |
| Clasificación | AUC macro | **0.9719** |

> **Segmentación:** U-Net/ResNet34 entrenado con split 80/20, loss Dice+BCE, AdamW lr=1e-4, early stopping paciencia=5. Convergencia en época 7, early stopping en época 12. Modelo publicado en [HF Hub](https://huggingface.co/Jesusrodriguezf90/unet-resnet34-isic2018-segmentation).

> **Clasificación:** EfficientNet-B0 entrenado con fine-tuning progresivo (5 épocas encoder congelado + 30 descongelado), WeightedRandomSampler + CrossEntropyLoss con pesos de clase para gestionar el desbalance severo (ratio 58.3x). Tres experimentos realizados — la configuración final usa `epochs_unfrozen=30` sin dropout adicional ni label smoothing, que demostró mejor generalización. Modelo publicado en [HF Hub](https://huggingface.co/Jesusrodriguezf90/efficientnet-b0-isic2018-classification).

### Evaluación del pipeline completo

La evaluación encadena U-Net → EfficientNet sobre las 10.015 imágenes de HAM10000 usando pseudo-máscaras generadas por U-Net. Se comparan dos modos:

| Modo | F1 macro | AUC macro |
|---|---|---|
| A — sin segmentación (imagen completa) | 0.8411 | 0.9920 |
| B — con pseudo-máscara U-Net (pipeline real) | 0.3554 | 0.8288 |
| **ΔB - A** | **-0.4858** | **-0.1632** |

> **Análisis crítico:** la segmentación previa degrada el rendimiento del clasificador. Hay dos causas identificadas. Primera: *distribution shift* — U-Net se entrenó con las imágenes de Task 1 y genera pseudo-máscaras sobre Task 3, una distribución que no vio durante el entrenamiento, resultando en máscaras de menor calidad. Segunda: EfficientNet aprendió a clasificar usando también el contexto de piel sana circundante — eliminarlo con la máscara destruye información diagnóstica relevante. Este resultado es consistente con Hasan et al. (2020), que reporta que eliminar completamente el fondo degrada el rendimiento de clasificación. La clase más perjudicada es DF (dermatofibroma, Δ-0.635) por ser una lesión pequeña donde el contexto circundante es especialmente relevante.

---

## Proceso experimental — Clasificación

El entrenamiento de EfficientNet-B0 requirió tres experimentos para determinar la configuración óptima:

| Experimento | Cambios respecto al anterior | F1 macro | AUC |
|---|---|---|---|
| E1 — baseline | `epochs_unfrozen=20` | 0.7493 | 0.9704 |
| E2 — regularización | + `drop_rate=0.2` + `label_smoothing=0.1` + `epochs_unfrozen=30` | ~0.50 | ~0.96 |
| E3 — configuración final ✓ | Solo `epochs_unfrozen=30` | **0.7700** | **0.9719** |

E2 demostró que combinar dropout adicional y label smoothing con un dataset fuertemente desbalanceado ralentiza la convergencia sin beneficio neto. E3 confirmó que el único cambio necesario era ampliar el margen de entrenamiento para que el early stopping pudiera actuar correctamente.

---

## Datos de prueba

> El dataset utilizado para el desarrollo y evaluación del pipeline es el benchmark de referencia mundial para análisis automatizado de lesiones cutáneas dermoscópicas.

**ISIC 2018 — Skin Lesion Analysis Towards Melanoma Detection**  
International Skin Imaging Collaboration (ISIC)  
🔗 https://challenge.isic-archive.com/data/#2018

| Dataset | Imágenes | Máscaras | Etiquetas | Uso |
|---|---|---|---|---|
| Task 1 (entrenamiento) | 2.594 | ✅ | ❌ | Entrena U-Net |
| Task 3 / HAM10000 | 10.015 | ❌ | ✅ | Entrena EfficientNet + evaluación pipeline |
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
| ✅ | Segmentación U-Net (notebook 02_segmentation) |
| ✅ | Publicación del modelo de segmentación en HF Hub |
| ✅ | Clasificación EfficientNet (notebook 03_classification) |
| ✅ | Publicación del modelo de clasificación en HF Hub |
| ✅ | Evaluación del pipeline completo (notebook 04_evaluation) |
| 🔲 | Demo Gradio en HF Spaces |

---

## Limitaciones

- El modelo está entrenado con imágenes **dermoscópicas estándar** capturadas con dermatoscopio — no es aplicable a fotografías clínicas convencionales ni fotos de smartphone sin dermatoscopio
- La escala física de la lesión en milímetros no se usa como feature — el modelo aprende patrones de textura, color y forma relativos a la imagen
- El dataset ISIC 2018 está fuertemente desbalanceado — NV (nevus) representa el 66.9% de Task 3, mientras que DF (dermatofibroma) es solo el 1.1%
- Task 1 y Task 3 son conjuntos disjuntos — U-Net y EfficientNet se entrenan con datos independientes sin solapamiento
- La evaluación del pipeline encadenado usa pseudo-máscaras de U-Net sobre Task 3 — U-Net fue entrenado con Task 1, por lo que existe distribution shift entre los datos de entrenamiento de segmentación y los de evaluación
- MEL (melanoma) presenta precision 0.40 con recall 0.82 en clasificación individual — el modelo genera falsos positivos de melanoma, comportamiento clínicamente aceptable pero que refleja la dificultad de distinguir melanoma de nevus en imágenes ambiguas
- La segmentación previa degrada la clasificación en el pipeline encadenado (ΔF1 -0.486) — EfficientNet usa el contexto de piel circundante como información diagnóstica, que se pierde al enmascarar el fondo

---

## Referencias

- Codella et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC). *arXiv:1902.03368*
- Tschandl et al. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*
- Hasan et al. (2020). The Effects of Skin Lesion Segmentation on the Performance of Dermatoscopic Image Classification. *arXiv:2008.12602*

---

## Licencia

Este proyecto está distribuido bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  Construido para la medicina y la inteligencia artificial
</p>
