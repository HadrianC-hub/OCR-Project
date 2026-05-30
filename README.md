# OCR-Project

Sistema de digitalización documental para textos impresos en español, desarrollado como Trabajo de Diploma en colaboración con el Instituto Cubano de Investigación Cultural Juan Marinello. El proyecto cubre el ciclo completo de la cadena OCR: generación de un *corpus* sintético, preprocesado de páginas digitalizadas, entrenamiento y evaluación de un modelo CRNN+CTC, y una aplicación web Django que integra todo en un flujo operativo de subida, transcripción, revisión y exportación.

---

## Contenido del repositorio

```
OCR-Project/
├── README.md                    ← este archivo
│
├── Code/                        ← código fuente de los cuatro subsistemas
│   ├── Data_Generator/          ← generación del corpus sintético
│   ├── Image_Preprocessing/     ← pipeline de preprocesado de imágenes
│   ├── Model/                   ← modelo CRNN+CTC (entrenamiento e inferencia)
│   ├── Other_Scripts/           ← utilidades auxiliares
│   └── Web_Page/                ← aplicación web Django
│
├── Thesis/                      ← PDF compilado de la tesis
│   └── Thesis.pdf
│
└── Thesis_LATEX/                ← fuentes LaTeX de la tesis (recompilable)
    └── ...
```

Los cuatro subsistemas dentro de `Code/` se corresponden, uno a uno, con los cuatro capítulos técnicos de la tesis. Cada uno se puede usar de forma independiente, aunque el flujo natural los encadena: el generador produce el *corpus* sobre el que el modelo entrena, el preprocesado prepara las líneas que el modelo transcribe, y la aplicación web articula el modelo y el preprocesado en una herramienta accesible.

---

## Ruta rápida según lo que quieras hacer

| Si quieres… | Ve a |
|-------------|------|
| Leer la tesis | `Thesis/Thesis.pdf` |
| Recompilar la tesis | `Thesis_LATEX/` (instrucciones más abajo) |
| Levantar la aplicación web y usarla | `Code/Web_Page/ocr_project/` |
| Reentrenar el modelo OCR | `Code/Model/` |
| Regenerar el *corpus* sintético | `Code/Data_Generator/` |
| Probar el preprocesado sobre tus propias imágenes | `Code/Image_Preprocessing/` |
| Comparar greedy vs *beam*+LM sobre líneas reales | `Code/Other_Scripts/compare.py` |

---

## 1. `Code/Data_Generator/` — generación del *corpus* sintético

Construye los pares imagen–transcripción que alimentan el entrenamiento del modelo. Toma un *corpus* textual en español, muestrea cadenas con una distribución de frecuencias de caracteres corregida para reducir el desequilibrio entre clases, y las renderiza con quince fuentes tipográficas distintas más efectos sintéticos de máquina de escribir y degradación de escaneado.

### Estructura

```
Data_Generator/
├── Corpus/                          ← textos fuente (Project Gutenberg)
│   ├── libro_2000.txt, libro_13507.txt, libro_24536.txt, libro_49836.txt
│   └── Augmented_corpus.txt         ← versión consolidada y aumentada
├── Fonts/                           ← 15 fuentes .ttf/.otf
├── download_corpus.py               ← descarga libros de Project Gutenberg
├── generator.py                     ← renderiza las imágenes
├── vocabulary_generator.py          ← construye el vocab.txt indexado
└── Exploratory_Data_Analysis.py     ← genera el dashboard EDA
```

### Cómo se usa

1. **Descarga el *corpus* textual** (solo si quieres regenerarlo desde cero):
   ```bash
   cd Code/Data_Generator
   python download_corpus.py
   ```
   Descarga los cuatro libros del listado `LIBROS_IDS` de Project Gutenberg, los limpia (quita cabeceras y caracteres no españoles) y los guarda en `Corpus/`.

2. **Construye el vocabulario** indexado de 100 símbolos:
   ```bash
   python vocabulary_generator.py
   ```
   Produce `Vocabulary/vocab.txt`, ordenado por frecuencia descendente dentro de cada grupo (minúsculas → mayúsculas → dígitos → puntuación).

3. **Genera el *corpus* de imágenes**:
   ```bash
   python generator.py
   ```
   Parámetros principales editables en la cabecera del fichero:
   - `TOTAL_IMAGES` = número de imágenes por fuente (por defecto 2000)
   - `FONT_SIZE`, `MARGIN_*` = parámetros tipográficos
   - `FRECUENCIAS_OBJETIVO` = distribución de frecuencias de caracteres del español, corregida para mitigar el desequilibrio entre clases

   La salida queda en `Generated/`, con la estructura `Generated/<fuente>/<imagen>.png` y un fichero `labels.json` por fuente con las transcripciones.

4. **Analiza estadísticamente el *corpus* generado**:
   ```bash
   python Exploratory_Data_Analysis.py
   ```
   Produce `EDA/eda_dashboard.png` (ocho paneles con frecuencias observadas vs objetivo, divergencias, distribución de longitudes, etc.) y `EDA/eda_informe.txt`.

---

## 2. `Code/Image_Preprocessing/` — *pipeline* de preprocesado

Convierte una página digitalizada (o una línea ya recortada) en el formato exacto que el modelo OCR espera como entrada: imagen en escala de grises, normalizada a 64 px de alto, binarizada de forma adaptativa, con la inclinación corregida y la línea aislada del resto del facsímil.

### Estructura

```
Image_Preprocessing/
├── visualize.py                     ← punto de entrada con visualizaciones
└── preprocessing/                   ← módulos del pipeline
    ├── binarization.py              ← binarización adaptativa + limpieza
    ├── config.py                    ← dataclasses con parámetros y métricas
    ├── line_preprocess.py           ← preprocesado de líneas ya recortadas
    ├── line_processing.py           ← detección y normalización de líneas
    └── pipeline.py                  ← orquestador del pipeline multilínea
```

### Cómo se usa

Edita la cabecera de `visualize.py` para seleccionar las imágenes que quieres procesar y el modo de operación:

- `SINGLE_LINE_MODE = False` (por defecto): procesa páginas completas y aplica el *pipeline* multilínea (binarización → corrección de inclinación → detección de bloques → segmentación de líneas).
- `SINGLE_LINE_MODE = True`: trata cada imagen como una línea ya recortada y aplica solo la normalización para inferencia.

Ejecuta:
```bash
cd Code/Image_Preprocessing
python visualize.py
```

Genera para cada imagen un mosaico con todas las etapas intermedias (binarización, máscara de encuadernación, detección de líneas, segmentación, normalización final), útil para ajustar parámetros sobre un nuevo tipo de documento.

---

## 3. `Code/Model/` — modelo CRNN+CTC

Implementación del modelo de reconocimiento: arquitectura CRNN (CNN convolucional + BiLSTM), función de pérdida CTC, generador de datos con muestreo por agrupamiento (*bucketing*) para anchos variables, decodificación voraz (*greedy*) y con búsqueda en haz, métricas (CER, WER, exactitud por línea), intervalos de confianza por *bootstrap* y validación cruzada estratificada por fuente tipográfica.

### Estructura

```
Model/
├── model.py     ← arquitectura CRNN
├── dataset.py   ← OCRDataset, NoAugSubset, BucketBatchSampler, decodificadores
├── metrics.py   ← CER/WER/LineAcc + bootstrap + Kruskal-Wallis por fuente
├── train.py     ← orquestador de entrenamiento, validación, fine-tuning, CV, LOFO
└── infer.py     ← inferencia sobre imágenes locales (uso de línea de comandos)
```

### Cómo se usa

Este código está pensado para **ejecutarse en Kaggle** (GPU T4 o P100), donde los conjuntos de datos se montan como *Kaggle datasets*. Las rutas en `train.py` apuntan por defecto a `/kaggle/input/` y `/kaggle/working/`. Si quieres ejecutarlo localmente, edita el bloque `CONFIG` al principio de `train.py`.

#### Entrenamiento desde cero

Edita `CONFIG["mode"] = "train"` en `train.py` y ejecuta:
```bash
python train.py
```
Configurable: número de épocas, tasa de aprendizaje, tamaño de lote, política de aumentación, paciencia para *early stopping*, etc.

#### *Fine-tuning* a partir de un *checkpoint*

`CONFIG["mode"] = "fine_tune"`, ajusta `CONFIG["resume_from"]` con la ruta del *checkpoint*, y ejecuta. La etapa de *fine-tuning* continúa el entrenamiento sobre el repertorio tipográfico ampliado.

#### Validación cruzada estratificada por fuente

`CONFIG["mode"] = "cross_validation"` ejecuta una *5-fold cross-validation* con particiones estratificadas por fuente tipográfica, y reporta media e intervalos de confianza por *bootstrap* sobre cada métrica.

#### *Leave-One-Font-Out*

`CONFIG["mode"] = "leave_one_font_out"` entrena un modelo por cada fuente del repertorio dejándola fuera del entrenamiento y evaluando sobre ella, para medir la generalización a tipografías no vistas.

#### Inferencia sobre imágenes locales

Para transcribir una imagen sin tocar `train.py`:
```bash
python infer.py --checkpoint best_model.pt --image mi_linea.png
```
Opciones: `--beam-width`, `--lm-path` (ruta a un `.arpa`), `--lm-alpha`.

---

## 4. `Code/Web_Page/ocr_project/` — aplicación web Django

Aplicación web completa que integra los componentes anteriores en una herramienta operativa para el personal del centro Juan Marinello: subida de facsímiles, ejecución asíncrona del OCR, edición y revisión de transcripciones sobre el facsímil, gestión de documentos y usuarios, búsqueda híbrida por contenido y exportación a PDF y EPUB.

### Estructura

```
Web_Page/ocr_project/
├── README.md                        ← guía específica de la aplicación (léela)
├── manage.py                        ← punto de entrada Django
├── install_dependencies.py          ← instalador asistido
├── requirements.txt                 ← dependencias Python
├── .env                             ← variables de entorno (sin credenciales)
├── ocr_predict.py                   ← módulo de inferencia OCR completo
│
├── ocr_project/                     ← configuración Django (settings, urls)
├── apps/
│   ├── accounts/                    ← usuarios, roles, autenticación
│   ├── documents/                   ← documentos, páginas, transcripciones, regiones
│   ├── ocr/                         ← motor OCR, corrección ortográfica, segmentación
│   ├── search/                      ← búsqueda híbrida BM25 + embeddings semánticos
│   └── stats/                       ← métricas de uso y auditoría
├── preprocessing/                   ← copia operativa del pipeline de preprocesado
├── models/                          ← carpeta donde colocar los pesos (ver README interno)
├── docs/
│   ├── transcript_format.md         ← formato del XML de transcripción
│   └── transcript_v1.xsd            ← esquema XSD del formato
├── templates/                       ← plantillas HTML
├── static/                          ← CSS y JS
└── media/                           ← facsímiles, transcripciones y caché en disco
```

### Cómo se levanta

El `README.md` que vive dentro de `Web_Page/ocr_project/` contiene la guía completa de instalación, configuración y uso. En resumen:

1. **Requisitos**: Python 3.10+, PostgreSQL 13+, 8 GB de RAM mínimo.

2. **Entorno virtual y dependencias**:
   ```bash
   cd Code/Web_Page/ocr_project
   python -m venv venv
   source venv/bin/activate            # En Windows: venv\Scripts\activate
   python install_dependencies.py
   ```

3. **Configura `.env`** con la `SECRET_KEY` de Django y la `DATABASE_URL` de tu PostgreSQL.

4. **Migra la base de datos** y **crea el primer superadministrador**:
   ```bash
   python manage.py migrate
   python manage.py create_superadmin
   ```

5. **Coloca los modelos** en `models/`:
   - `models/printed/best_model.pt` — modelo OCR impreso (CRNN+CTC) producido por `Code/Model/`
   - `models/vocab.txt` — vocabulario indexado del modelo
   - `models/kenLM.arpa` — modelo de lenguaje (opcional, mejora la decodificación sobre documentos reales)
   - (Opcional) `models/trocr_es_finetuned/` para manuscritos, `models/beto/` para *reranker* ortográfico

6. **Arranca el servidor**:
   ```bash
   python manage.py runserver
   ```
   La aplicación queda disponible en `http://localhost:8000`.

### Roles de usuario

- **Administrador principal**: gestión completa del sistema, incluida la creación de administradores.
- **Administrador**: gestiona usuarios *workers* y todos los documentos.
- **Worker**: sube documentos, ejecuta OCR, edita y revisa transcripciones.

### Formato de transcripción

La aplicación guarda las transcripciones como XML siguiendo el formato `transcript_v1` descrito en `docs/transcript_format.md`. Cada página genera un fichero `transcripts/{document_id}/page_{order:03d}.xml` legible y editable a mano si hace falta. El esquema XSD está en `docs/transcript_v1.xsd`.

---

## 5. `Code/Other_Scripts/` — utilidades auxiliares

Scripts sueltos que no pertenecen a ningún subsistema concreto pero resultan útiles para tareas específicas alrededor del proyecto.

### `compare.py` — comparación cualitativa greedy vs *beam*+LM

Reproduce exactamente la pila de inferencia de la aplicación web (incluido el lector `ArpaLM` en Python puro, sin necesidad de `kenlm`) y compara las dos decodificaciones sobre imágenes de líneas locales. Pensado para construir ejemplos visuales para informes o presentaciones.

Estructura de trabajo:
```
ocr-comparison/
├── compare.py
├── ocr_predict.py        ← copia desde Code/Web_Page/ocr_project/
├── models/
│   ├── best_model.pt
│   ├── vocab.txt
│   └── kenLM.arpa
└── images/               ← imágenes de líneas a comparar
```

Uso:
```bash
python compare.py --output resultado.txt
```

Para cada imagen produce dos transcripciones (greedy y beam+LM) y un resumen del porcentaje de imágenes en las que los decodificadores difieren.

### `convert_beto_to_onnx.py` — conversión del *reranker* BETO

Descarga BETO de HuggingFace, lo exporta a ONNX, lo cuantiza a int8 (~110 MB finales) y verifica que `onnxruntime` puede ejecutarlo. Ejecútalo una sola vez en una máquina con acceso a internet y suficiente espacio:

```bash
pip install transformers torch onnxruntime onnxscript
python convert_beto_to_onnx.py --output models/beto --quantize
```

El directorio resultante se copia tal cual al servidor de producción en `Web_Page/ocr_project/models/beto/`. A partir de ahí ya no hace falta `torch` para usar el *reranker*.

### `test_inference_kaggle.ipynb` — evaluación sobre el conjunto de prueba

*Notebook* para Kaggle que monta el conjunto de prueba (1995 imágenes, 15 fuentes), carga el *checkpoint* del modelo y produce el informe estadístico completo: CER/WER/exactitud por línea, intervalos de confianza por *bootstrap*, y test de Kruskal-Wallis para comparar el rendimiento por fuente tipográfica.

Antes de ejecutarlo, edita la celda de **Configuración de rutas** con los identificadores de tus *Kaggle datasets*.

---

## 6. `Thesis/` — documento de la tesis

Contiene el PDF compilado del Trabajo de Diploma:

```
Thesis/
└── Thesis.pdf
```

Es la versión final lista para imprimir y leer. No requiere ninguna acción adicional: ábrelo con tu lector de PDF habitual.

---

## 7. `Thesis_LATEX/` — fuentes LaTeX de la tesis

Código fuente completo de la tesis en LaTeX, organizado según la plantilla `uhthesis` de la Universidad de La Habana. Permite regenerar el PDF, modificar el documento, o reutilizar la estructura para otros trabajos.

### Estructura

```
Thesis_LATEX/
├── Thesis.tex                       ← documento maestro
├── uhthesis.cls                     ← plantilla de la Universidad de La Habana
├── Bibliography.bib                 ← bibliografía en BibLaTeX
│
├── FrontMatter/                     ← portada, resumen, dedicatoria, agradecimientos
├── MainMatter/                      ← seis capítulos técnicos
│   ├── Introduction.tex
│   ├── Background.tex
│   ├── Corpus.tex
│   ├── Preprocessing.tex
│   ├── Model.tex
│   └── WebApplication.tex
├── BackMatter/                      ← conclusiones, recomendaciones, bibliografía
└── Graphics/                        ← figuras (PNG/JPG/PDF/SVG)
```

### Cómo se compila

Requisitos: una distribución TeX Live completa o equivalente (MiKTeX, TeXmacs) con `pdflatex`, `biber` y los paquetes habituales (`subcaption`, `algorithm2e`, `amsmath`, `xcolor`, `booktabs`, `tabularx`, `listings`).

Compilación completa desde la raíz:

```bash
cd Thesis_LATEX
pdflatex Thesis.tex
biber Thesis
pdflatex Thesis.tex
pdflatex Thesis.tex
```

(Las dos pasadas finales de `pdflatex` resuelven referencias cruzadas y la tabla de contenidos.)

El PDF resultante queda como `Thesis.pdf` en la misma carpeta. Puedes copiarlo a `Thesis/` para actualizar la versión publicada.

---

## Reconocimiento

Trabajo de Diploma desarrollado en la Facultad de Matemática y Computación de la Universidad de La Habana, en colaboración con el Instituto Cubano de Investigación Cultural Juan Marinello, que aportó el contexto institucional, los documentos reales para evaluación cualitativa y el entorno de despliegue del sistema.
