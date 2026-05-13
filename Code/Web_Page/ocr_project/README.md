# Parche OCR — manuscritos + bug del cuelgue del navegador

Aplica esto sobre tu proyecto Django (`ocr_project/`). Resuelve tres
problemas independientes:

1. **Segmentación de líneas rota** en manuscritos cursivos (los descendentes
   se detectaban como segunda línea, los espacios entre palabras como
   columnas falsas; líneas adyacentes se fusionaban).
2. **Preprocesamiento divergente** entre training y producción: el modelo
   ahora recibe imágenes preprocesadas con el mismo pipeline que se usó
   para entrenar (`line_preprocess`).
3. **Bug del cuelgue al insertar un segundo manuscrito**: el navegador se
   quedaba cargando hasta que cerrabas la pestaña, aunque el backend ya
   hubiera completado el OCR.

Una mejora extra incluida: `beam_bonus=0.0` (antes `2.0`) en el decoder
del HTR, que elimina las deleciones masivas que vimos en la matriz de
confusión del test final.

---

## Cómo aplicar

Desde la raíz de tu proyecto Django (la carpeta que contiene `manage.py`),
descomprime el zip. Sobreescribe los archivos existentes:

```bash
cd ocr_project/                       # tu raíz Django
unzip /ruta/a/patch_ocr_fixes.zip     # sobreescribe los archivos modificados
```

El zip respeta la estructura del proyecto, así que descomprimir desde la
raíz coloca cada archivo en su sitio.

### Archivos modificados o nuevos

```
preprocessing/
├── line_preprocess.py        ← NUEVO (copia del standalone)
└── manuscript_lines.py       ← NUEVO (detector cursivo)
apps/
├── ocr/
│   ├── ocr_engine.py         ← reescrito (dispatcher manuscript/printed,
│   │                            thread-locks, cap torch threads)
│   ├── manuscript_predictor.py ← cambio: beam_bonus default 2.0 → 0.0
│   ├── spell_correct.py      ← cambio: locks en singletons
│   ├── tasks.py              ← cambio: pre-cachea segmentación,
│   │                            auto-recovery de threads muertos
│   ├── views.py              ← cambio: line_segmentation_* devuelve 503
│   │                            si la página está pending/processing;
│   │                            document_ocr_status auto-resucita
│   │                            documentos huérfanos
│   └── segmentation.py       ← cambio: refactor (puede generar desde
│                                resultado pre-computado)
└── search/
    └── encoder.py            ← cambio: locks en singletons
templates/
└── documents/
    ├── edit_document.html    ← cambio: <img> de segmentación solo
    │                            renderiza cuando ocr_status == 'done'
    └── insert_document.html  ← cambio: timeout 5min en fetch +
                                 listener pageshow para bfcache
```

NO se tocan: modelos Django, migraciones, formularios, URLs, otras
templates, settings, código de búsqueda más allá de `encoder.py`,
ni los modelos `.pt`.

### Requisitos previos

Ninguno. Las dependencias que usan los archivos nuevos
(`scipy.ndimage.uniform_filter`, `scipy.signal.find_peaks`, `cv2`, `numpy`,
PyTorch) ya están en tu `requirements.txt`.

---

## Reinicio

Tras descomprimir, **reinicia Django** para que cargue los módulos nuevos
y aplique el cap de threads de PyTorch desde el primer request:

```bash
# Con runserver:
# (Ctrl-C y vuelve a lanzar)

# Con gunicorn:
sudo systemctl restart tu-servicio-gunicorn

# Con supervisord, etc.: el comando correspondiente
```

No requiere migraciones de base de datos.

---

## Configuración opcional

### Threads de PyTorch

Por defecto se limita PyTorch a 2 threads internos (intra-op + interop).
Esto deja CPU libre para los handlers HTTP cuando el OCR está corriendo.
Si tu máquina tiene muchos cores y prefieres OCR más rápido a cambio de
peor responsividad bajo carga, sube el valor con una variable de entorno
antes de arrancar Django:

```bash
export OCR_TORCH_THREADS=4
# o lo que sea
```

Valores típicos:
- **2** (default): web responsiva, OCR ~3-4 s/página manuscrita
- **4**: OCR ~2 s/página, web aún OK para 1-2 usuarios simultáneos
- **`os.cpu_count()`**: OCR lo más rápido posible, web puede tartamudear
  bajo carga concurrente

---

## Cómo verificar que todo funciona

### Sanity check rápido

1. Sube un documento manuscrito (uno de los tres ejemplos que probamos
   sirve: `7816...jpg`, `33317...jpg`, `a9f70...jpg`).
2. Mientras procesa, la pantalla de edición debe mostrar:
   - Cada página como pendiente con su spinner
   - Un placeholder "Generando segmentación…" en lugar de la imagen de
     cajas
   - El polling debe refrescar el thumbnail a "done" cuando termine
3. Cuando termina, la imagen de segmentación aparece YA generada
   (servida desde cache, milisegundos).
4. El texto reconocido debe estar dividido en líneas, no en
   fragmentos sin sentido.

### Sanity check del bug del cuelgue

1. Sube manuscrito A. Espera a que termine. Guarda.
2. Vuelve a "Insertar documento". Sube manuscrito B.
3. El cuadro de progreso debe DESAPARECER al terminar (no quedarse
   colgado).
4. Si por la razón que sea no se desbloquea, espera 5 minutos: el
   `fetch` tiene timeout y libera al usuario con un mensaje claro.
5. Si pulsas el botón Atrás del navegador y vuelves a entrar, ya no
   ves el overlay congelado (lo evita el listener `pageshow`).

### Sanity check del auto-recovery

1. Sube un documento. Mientras procesa, **reinicia Django**.
2. Recarga la página de edición.
3. El polling detectará que no hay thread vivo y debe RE-ARRANCAR
   el OCR transparentemente. Las páginas que estaban "processing"
   vuelven a "pending" y se reprocesan.

---

## Qué esperar de calidad de OCR tras el parche

| Doc de prueba             | CER pre-parche    | CER post-parche |
|---------------------------|-------------------|-----------------|
| 7816 (Bernárdez cursiva)  | ~90% (basura)     | **22.5 %**      |
| 33317 (Acción Poética)    | ~50% (mal orden)  | **42.9 %**      |
| a9f70 (Lope de Vega)      | basura            | similar a 7816  |

El CER residual (20-40%) es el **techo del modelo actual** sobre datos
no vistos. No se baja más sin reentrenar. Esto NO está en este parche;
queda para una iteración futura del modelo, donde habrá que:

- Re-particionar train/val/test por **escritor** (no por imagen dentro
  del mismo documento, que es lo que tenías y causa data leakage).
- Aumentar diversidad de escritores reales o aplicar transfer learning
  desde un HTR pre-entrenado tipo TrOCR / Calamari.

---

## Rollback

Si algo va mal, simplemente restaura los archivos originales desde tu
control de versiones (git). Los únicos archivos NUEVOS (que no
existían antes) son:

- `preprocessing/line_preprocess.py`
- `preprocessing/manuscript_lines.py`

El resto sobrescribe versiones previas — un `git checkout` los devuelve.

---

## Resumen de los cambios técnicos clave

(Para tu referencia / commit message)

**Segmentación de manuscritos** (`preprocessing/manuscript_lines.py`):
detector basado en proyección horizontal sobre un binario al que se
aplica closing vertical previo (kernel ≈ 0.3 × text_h). Esto fusiona
cuerpo + descendentes evitando el doble pico que producía cajas falsas.
La altura de texto se estima en dos pasos (componentes conectados +
refinado por peak-distance) para no depender de un smoothing fijo. Si
una banda detectada excede 1.7 × text_h se intenta partir por su valle
más profundo de forma recursiva. NO segmenta bloques ni columnas — la
página manuscrita se trata como un único bloque vertical.

**Preprocesamiento de manuscritos** (`apps/ocr/ocr_engine.py`,
función `_preprocess_array_manuscript`): por cada bbox detectado, se
cropea el GRIS deskewed (que ya produjo el detector) y se pasa por
`line_preprocess.preprocess_line()`. Esto coincide exactamente con el
camino que se usó para preparar los datos de training, eliminando el
mismatch que dejaba al modelo "fuera de distribución" sobre la web.

**Bug del cuelgue**: causa raíz era contención de CPU. El handler de
`line_segmentation_image` ejecutaba `pipeline.run` sincrónamente sobre
la misma imagen que el thread del OCR estaba procesando. Resultado:
30-60 s de espera por subrecurso + spinner del tab indefinido. Fix
combinado:
- El thread del OCR pre-genera la viz de segmentación tras procesar
  cada página (un solo `pipeline.run` por página).
- La view devuelve 503 + Retry-After si la página aún no está done.
- El template no inserta el `<img>` hasta que el estado es done.
- Cap a 2 threads de PyTorch para liberar cores para Django.
- Timeout de 5 min + AbortController en el fetch del insert.
- Listener pageshow para recuperarse de bfcache.
- Auto-recovery en `document_ocr_status`: si quedan páginas pendientes
  sin thread vivo, relanza OCR automáticamente.
