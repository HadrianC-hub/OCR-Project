from pathlib import Path
from collections import Counter
from generator import FRECUENCIAS_OBJETIVO

CORPUS_DIR = Path("Corpus")
OUTPUT_DIR = Path("Vocabulary")
OUTPUT_FILE = OUTPUT_DIR / "vocab.txt"

#  CONSTRUCCIÓN DEL VOCABULARIO
def construir_vocab() -> list:
    """
    Construye la lista ordenada de caracteres del vocabulario.
    Retorna una lista donde índice i = carácter i del modelo.

    Orden:
      0          -> espacio (convención frameworks OCR)
      1 … a      -> letras minúsculas, por frecuencia descendente
      a+1 … b    -> letras mayúsculas, por frecuencia descendente
      b+1 … b+10 -> dígitos 0–9
      b+11 … N   -> puntuación y símbolos, por frecuencia descendente
    """
    minusculas = sorted(
        [c for c in FRECUENCIAS_OBJETIVO if c.islower() and c.isalpha()],
        key=lambda c: FRECUENCIAS_OBJETIVO[c], reverse=True,
    )
    mayusculas = sorted(
        [c for c in FRECUENCIAS_OBJETIVO if c.isupper() and c.isalpha()],
        key=lambda c: FRECUENCIAS_OBJETIVO[c], reverse=True,
    )
    digitos    = [str(d) for d in range(10)]
    puntuacion = sorted(
        [c for c in FRECUENCIAS_OBJETIVO if not c.isalpha() and not c.isdigit()],
        key=lambda c: FRECUENCIAS_OBJETIVO[c], reverse=True,
    )

    vocab = [" "] + minusculas + mayusculas + digitos + puntuacion

    # Eliminar duplicados manteniendo orden
    seen  = set()
    vocab = [c for c in vocab if not (c in seen or seen.add(c))]
    return vocab

#  GUARDADO
def guardar_vocab(vocab: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for char in vocab:
            f.write(char + "\n")
    print(f"vocab.txt guardado -> {OUTPUT_FILE.resolve()}")
    print(f"   {len(vocab)} caracteres  (índices 0 - {len(vocab) - 1})")
    print(f"   Token CTC 'blank' = índice {len(vocab)}  (añadido por el framework)")

#  TABLA DE ÍNDICES
def imprimir_tabla(vocab: list):
    print("\n── Tabla de índices ──────────────────────────────────────────────")
    print(f"  {'IDX':>4}  {'CHAR':^6}  CATEGORÍA")
    print(f"  {'─'*4}  {'─'*6}  {'─'*20}")
    for i, c in enumerate(vocab):
        if   c == " ":              cat = "espacio"
        elif c.islower() and c.isalpha(): cat = "minúscula"
        elif c.isupper() and c.isalpha(): cat = "mayúscula"
        elif c.isdigit():           cat = "dígito"
        else:                       cat = "puntuación / símbolo"
        display = "' '" if c == " " else c
        print(f"  {i:>4}  {display:^6}  {cat}")
    print(f"  {len(vocab):>4}  {'[blank]':^6}  token CTC (interno al framework)")
    print()

#  VERIFICACIÓN CONTRA EL CORPUS
def verificar_contra_corpus(vocab: list):
    """
    Lee todos los .txt de CORPUS_DIR y comprueba que cada carácter
    que aparece en ellos esté cubierto por el vocabulario.
    """
    if not CORPUS_DIR.exists():
        print(f"Carpeta '{CORPUS_DIR}' no encontrada - verificación omitida.")
        return

    archivos = sorted(CORPUS_DIR.glob("*.txt"))
    if not archivos:
        print(f"No se encontraron .txt en '{CORPUS_DIR}' — verificación omitida.")
        return

    print(f"\n── Verificando contra {len(archivos)} archivos en '{CORPUS_DIR}/' ──")
    chars_corpus = Counter()
    for archivo in archivos:
        chars_corpus.update(archivo.read_text(encoding="utf-8"))

    vocab_set   = set(vocab)
    fuera_vocab = {
        c: n for c, n in chars_corpus.items()
        if c not in vocab_set and c.strip() and c.isprintable()
    }

    if fuera_vocab:
        print("AVISO - caracteres en el corpus NO cubiertos por el vocabulario:")
        for c, n in sorted(fuera_vocab.items(), key=lambda x: x[1], reverse=True):
            print(f"   '{c}'  U+{ord(c):04X}  ->  {n:,} apariciones")
        print("   -> Añádelos a FRECUENCIAS_OBJETIVO en generator.py")
    else:
        print("OK - todos los caracteres del corpus están en el vocabulario.")

    sin_uso = [c for c in vocab if c not in chars_corpus and c != " "]
    if sin_uso:
        print(f"\n{len(sin_uso)} caracteres del vocab sin apariciones en el corpus:")
        print(f"   {''.join(sin_uso)}")
        print("   -> El modelo los tendrá en su alfabeto pero sin ejemplos reales.")

#  MAIN
if __name__ == "__main__":
    print("══════════════════════════════════════════════════════════════")
    print("  GENERADOR DE VOCABULARIO OCR")
    print("══════════════════════════════════════════════════════════════\n")

    vocab = construir_vocab()

    imprimir_tabla(vocab)
    guardar_vocab(vocab)
    verificar_contra_corpus(vocab)

    letras = [c for c in vocab if c.isalpha()]
    digs   = [c for c in vocab if c.isdigit()]
    punts  = [c for c in vocab if not c.isalpha() and not c.isdigit()]

    print("\n── Resumen ───────────────────────────────────────────────────")
    print(f"  Letras     : {len(letras):>3}  ({''.join(letras)})")
    print(f"  Dígitos    : {len(digs):>3}  ({''.join(digs)})")
    print(f"  Puntuación : {len(punts):>3}  ({''.join(punts)})")
    print(f"  TOTAL      : {len(vocab):>3}  (+ 1 blank CTC = {len(vocab) + 1} nodos de salida)")
    print("══════════════════════════════════════════════════════════════")