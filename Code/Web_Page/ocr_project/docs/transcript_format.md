# Formato de transcripción — `transcript_v1`

Las transcripciones de cada página se guardan como XML en

```
{MEDIA_ROOT}/transcripts/{document_id}/page_{order:03d}.xml
```

Una página por fichero. La numeración usa cero-relleno a 3 dígitos
(`page_001.xml`, `page_002.xml`, …) para que el orden lexicográfico
coincida con el orden numérico.

## Diseño

El esquema busca tres objetivos:

1. **Lectura humana directa.** Un revisor puede abrir cualquier XML
   con un editor de texto y entender (y editar) el contenido sin
   herramientas auxiliares.
2. **Sin namespaces ni dependencias externas.** Se serializa con
   `xml.etree.ElementTree` (estándar, viene con Python). No hay
   necesidad de `lxml`.
3. **Convertible a estándares mayores** (ALTO, PAGE-XML, TEI) si el
   archivo crece y necesita interoperabilidad institucional. Las cajas
   de regiones ya están en píxeles del facsimilar original, que es
   también la convención de ALTO.

> **Estándares alternativos considerados.** Se valoraron ALTO XML
> (estándar de facto en bibliotecas y archivos), PAGE-XML
> (PRImA / Transkribus) y TEI. ALTO es el camino natural si en algún
> momento se quiere ingestar el archivo en un repositorio
> institucional o intercambiar datos con otros archivos digitales.
> Para esta primera iteración se eligió un esquema propio mucho más
> compacto. La migración hacia ALTO es directa porque la información
> que contiene es un subconjunto.

## Estructura

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Transcript version="1">

  <Document id="42">
    <Title>Don Quijote de la Mancha</Title>
    <Author>Miguel de Cervantes</Author>
    <Year>1605</Year>
    <Type>printed</Type>           <!-- 'printed' | 'manuscript' -->
  </Document>

  <Page order="3" facsimile="page_003.jpg"/>

  <Timestamps>
    <Created>2026-05-09T14:30:00+02:00</Created>
    <Modified>2026-05-09T14:35:12+02:00</Modified>
  </Timestamps>

  <!-- Opcional. Regiones definidas por el usuario en el editor.
       Coordenadas en píxeles del FACSIMILAR ORIGINAL (no deskewed).
       Si están presentes, "OCR de regiones" recorta sólo estas zonas.
       order define la secuencia en la que se procesan. -->
  <Regions>
    <Region id="r1" order="1" x="120" y="80"  width="900" height="220"/>
    <Region id="r2" order="2" x="120" y="320" width="900" height="180"/>
  </Regions>

  <!-- Cuerpo de la transcripción.
       Cada <Line> es un salto lógico de línea. Las líneas vacías
       se conservan como <Line/>. La representación texto plano
       es el join con '\n' del contenido de los <Line>. -->
  <Body>
    <Line>En un lugar de la Mancha, de cuyo nombre no quiero acordarme,</Line>
    <Line>no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero,</Line>
    <Line/>
    <Line>adarga antigua, rocín flaco y galgo corredor.</Line>
  </Body>
</Transcript>
```

### Reglas

- **Codificación**: siempre UTF-8 (caracteres acentuados se guardan
  literalmente, no como entidades).
- **Atributo `version`**: del elemento raíz. Permite evolución del
  esquema sin romper lectores antiguos.
- **`<Document id>`**: id numérico del documento Django. Sólo
  informativo; la fuente de verdad de los metadatos sigue siendo la
  base de datos.
- **`<Page order>`**: orden de la página dentro del documento.
- **Coordenadas de regiones**: enteros, en píxeles del facsimilar
  original, sin tener en cuenta el deskew. El pipeline de OCR aplica
  deskew local al recorte.
- **`<Body>`**: lista plana de líneas. Para reconstruir el texto
  plano, unir el contenido de los `<Line>` con `\n`.

### Lo que NO se guarda en este XML

- **Cajas de líneas detectadas automáticamente**. Son artefacto
  derivado del pipeline. Se cachean en
  `{MEDIA_ROOT}/segmentation/{document_id}/page_{order:03d}_lines.json`
  y se regeneran si el JPG es más viejo que el facsimilar. No tiene
  sentido guardarlas en el transcript: si el usuario edita el texto,
  el mapeo línea↔caja deja de tener significado.

- **Confidence scores del modelo**. ALTO los guarda; se podrían añadir
  fácilmente en una versión futura del esquema.

## API

Toda lectura/escritura pasa por `apps/documents/transcripts.py`:

| Función                                       | Descripción                                    |
|-----------------------------------------------|------------------------------------------------|
| `load(doc_id, order)`                         | `TranscriptData` o `None` si no existe         |
| `get_text(doc_id, order)`                     | texto plano (`'\n'.join(lines)`)               |
| `get_regions(doc_id, order)`                  | lista de `Region`                              |
| `save(TranscriptData)`                        | escritura atómica (`.tmp` → rename)            |
| `save_text(doc_id, order, text, **kwargs)`    | atajo: actualiza texto, conserva el resto      |
| `save_regions(doc_id, order, regions)`        | atajo: actualiza regiones, conserva el texto   |
| `delete_page_transcript(doc_id, order)`       | borra el fichero de una página                 |
| `delete_document_transcripts(doc_id)`         | borra el directorio entero del documento       |

Desde el modelo `Page` se accede via las propiedades:

```python
page.text                       # → str (lee del XML)
page.text = "nueva línea\n..."  # ← escribe en el XML
page.get_regions()              # → [Region, ...]
page.set_regions([Region(...)]) # ← persiste regiones
```

## Validación

Hay un XSD de referencia en `docs/transcript_v1.xsd` por si se quiere
validar con `xmllint --schema`. La aplicación no lo aplica en runtime
(el esquema es lo bastante simple como para que un parser permisivo
sea suficiente y nunca rompa por XML técnicamente "incorrectos" pero
recuperables).
