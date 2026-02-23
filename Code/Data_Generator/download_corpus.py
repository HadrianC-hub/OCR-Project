import requests
import re
import os

# Lista de IDs de Project Gutenberg en español
LIBROS_IDS = [2000, 13507, 24536, 49836] 

def limpiar_texto(texto):
    # Quitamos el encabezado y pie de página de Gutenberg (aproximado)
    start_marker = "START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "END OF THIS PROJECT GUTENBERG EBOOK"
    
    start_idx = texto.find(start_marker)
    end_idx = texto.rfind(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        texto = texto[start_idx + len(start_marker):end_idx]

    # Solo permitimos caracteres españoles y puntuación estándar
    # Letras, tildes, eñes, números y signos de puntuación
    patron = r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\s.,;:¡!¿?()\"\'\-\—\«\»\*\&\[\]\%\$\#]'
    texto_limpio = re.sub(patron, '', texto)
    
    # Normalizar espacios
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    return texto_limpio

# Crear carpeta de salida si no existe
if not os.path.exists('corpus_individuales'):
    os.makedirs('corpus_individuales')

for id in LIBROS_IDS:
    print(f"Procesando ID {id}...")
    url = f"https://www.gutenberg.org/files/{id}/{id}-0.txt"
    
    try:
        r = requests.get(url, timeout=15)
        r.encoding = 'utf-8'
        
        if r.status_code == 200:
            texto_final = limpiar_texto(r.text)
            
            nombre_archivo = f"Corpus/libro_{id}.txt"
            with open(nombre_archivo, "w", encoding="utf-8") as f:
                f.write(texto_final)
            print(f" -> [EXITO] Guardado como {nombre_archivo}")
        else:
            print(f" -> [ERROR] No se pudo acceder al ID {id} (Status: {r.status_code})")
            
    except Exception as e:
        print(f" -> [ERROR] Error con ID {id}: {e}")
