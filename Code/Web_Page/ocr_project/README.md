# ArchivoOCR

Plataforma web para digitalización y búsqueda de documentos manuscritos e impresos.
Permite subir facsímiles, transcribirlos automáticamente con OCR, corregir el
texto reconocido, organizar los documentos y buscarlos por contenido o metadatos.

## Tecnologías

- **Backend**: Django 4.2 + PostgreSQL
- **OCR impreso**: CRNN+CTC entrenado sobre líneas de texto impreso
- **OCR manuscrito**: TrOCR fine-tuneado sobre manuscritos cursivos en español
- **Búsqueda**: BM25 + embeddings semánticos multilingual-e5-small (rerank híbrido)
- **Corrección ortográfica**: SymSpell + reranker BERT (BETO) opcional

## Instalación

### Requisitos

- Python 3.10 o superior
- PostgreSQL 13 o superior
- 8 GB de RAM mínimo (16 GB recomendado para uso fluido con varios usuarios)

### Pasos

1. Clona el repositorio y entra al directorio del proyecto.

2. Crea un entorno virtual e instala las dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate    # En Windows: venv\Scripts\activate
   python install_dependencies.py
   ```

3. Configura las variables de entorno en `.env`:

   ```
   SECRET_KEY=tu_clave_secreta
   DATABASE_URL=postgres://user:password@localhost/archivo_ocr
   DEBUG=False
   ```

4. Aplica las migraciones de base de datos:

   ```bash
   python manage.py migrate
   ```

5. Crea el primer usuario administrador:

   ```bash
   python manage.py create_superadmin
   ```

6. Descarga los modelos y colócalos en la carpeta `models/`:

   - `models/printed/best_model.pt` — modelo OCR impreso
   - `models/trocr_es_finetuned/` — modelo TrOCR manuscrito
   - `models/multilingual-e5-small/` — embeddings para búsqueda semántica
   - `models/beto/` — reranker BERT opcional

7. Inicia el servidor:

   ```bash
   python manage.py runserver
   ```

   La aplicación queda disponible en `http://localhost:8000`.

## Roles de usuario

- **Administrador**: gestión total, incluye creación de usuarios.
- **Worker**: sube documentos, transcribe y corrige texto.
- **Lector**: solo consulta y búsqueda.

## Variables de entorno opcionales

- `OCR_TORCH_THREADS` — número de threads internos de PyTorch (por defecto `2`).
  Subir mejora la velocidad del OCR pero reduce la capacidad de respuesta del
  servidor con varios usuarios simultáneos.

- `OCR_TROCR_PATH` — ruta alternativa al modelo TrOCR si no se quiere usar la
  ubicación por defecto `models/trocr_es_finetuned/`.

## Estructura del proyecto

```
ocr_project/             Configuración Django
apps/
├── accounts/            Usuarios y autenticación
├── documents/           Documentos, páginas, transcripciones
├── ocr/                 Motores OCR y corrección ortográfica
├── search/              Indexado y búsqueda híbrida
└── stats/               Métricas de uso
preprocessing/           Pipeline de preprocesado de imágenes
templates/               Plantillas HTML
static/                  CSS y JS
docs/                    Formato del XML de transcripción
```

## Soporte

Para reportar problemas o solicitar funcionalidades, abre una issue en el
repositorio del proyecto.
