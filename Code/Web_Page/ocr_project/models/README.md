# Modelos OCR

Esta carpeta contiene los pesos de los modelos OCR.

## Estructura

```
models/
├── printed/
│   └── best_model.pt      ← Modelo CRNN+CTC para documentos impresos
├── manuscript/
│   └── best_model.pt      ← (Futuro) Modelo para manuscritos
└── kenLM.arpa             ← (Opcional) Modelo de lenguaje para beam search
```

## Dónde colocar best_model.pt

Copia tu fichero `best_model.pt` en:

    ocr_project/models/printed/best_model.pt

Reinicia el servidor Django después de colocarlo.
El modelo se cargará automáticamente la primera vez que se ejecute el OCR.

## Modelo de manuscritos

Cuando tengas el modelo para escritura manual, colócalo en:

    ocr_project/models/manuscript/best_model.pt

Se cargará automáticamente sin modificar código.

## Corrección ortográfica

La corrección post-OCR usa dos paquetes pip que traen sus datos dentro
del propio paquete — **no hay archivos para descargar manualmente**:

  - `symspellpy` → algoritmo rápido de corrección por distancia de edición.
  - `wordfreq`   → frecuencias del español (cubre formas verbales comunes).

Ambos están en `requirements.txt`. Solo asegúrate de instalar las
dependencias:

    pip install -r requirements.txt

El corrector se construye la primera vez que se procesa un documento
(~3 s, una sola vez por arranque del servidor) y queda cacheado en
memoria. Si las librerías no están instaladas, la corrección queda
deshabilitada en silencio (con un WARNING en logs) y el OCR sigue
funcionando sin tocar el texto.

Para desactivar la corrección manualmente, edita
`apps/ocr/spell_correct.py` y pon `SPELLCHECK_ENABLED = False`.
