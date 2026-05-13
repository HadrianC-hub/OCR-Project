"""
benchmark_spellcheck.py — Compara la precisión del corrector ortográfico
con y sin el reranker BERT (BETO) sobre texto español sintético con
errores OCR realistas.

Uso:
    python manage.py benchmark_spellcheck
    python manage.py benchmark_spellcheck --noise 0.15 --seed 7
    python manage.py benchmark_spellcheck --no-bert      # solo SymSpell
    python manage.py benchmark_spellcheck --corpus mi_archivo.txt

El benchmark:
  1. Toma un texto limpio (embebido o de fichero)
  2. Inyecta errores OCR realistas (sustituciones por confusiones
     visuales, borrados, inserciones)
  3. Aplica los tres modos de corrección:
        - sin corregir
        - SymSpell solo  (el corrector actual)
        - SymSpell + BETO  (el corrector nuevo)
  4. Mide WER (word error rate), CER (character error rate),
     número de cambios y tiempo de procesado
  5. Muestra una tabla comparativa

Sin GPU es lento: la primera carga de BETO tarda 30-60 s y luego cada
página son 1-5 s.
"""

import random
import re
import sys
import time
from pathlib import Path

from django.core.management.base import BaseCommand


# ── Texto de referencia embebido (~480 palabras de español moderno) ─────────
DEFAULT_CORPUS = """\
La pequeña librería estaba en una calle estrecha del barrio antiguo, entre una
panadería y una tienda de instrumentos musicales. Quien pasaba por delante sin
prestar atención podía no notarla, porque el escaparate era discreto y las
letras del rótulo, doradas sobre fondo verde oscuro, llevaban allí más de
medio siglo y habían perdido buena parte de su brillo original.

Dentro, sin embargo, el espacio era más grande de lo que sugería la fachada.
Había estanterías hasta el techo, mesas con novedades, una sección apartada
de libros antiguos, y al fondo una salita con dos sillones y una lámpara de
pie donde los clientes habituales podían sentarse a hojear sin que nadie los
molestara. El librero, un hombre de barba blanca y gafas pequeñas redondas,
conocía a casi todos por el nombre y recordaba con precisión qué autores les
gustaban, qué títulos habían comprado el año anterior y cuáles probablemente
les interesarían a continuación.

Por las tardes, especialmente los sábados, el establecimiento se llenaba de
estudiantes que venían a buscar manuales para la universidad, jubilados que
preferían los ensayos históricos y algún turista despistado que entraba para
preguntar por una guía de la ciudad. La conversación se mezclaba con el
sonido de las páginas al pasar y con el tic tac del reloj de pared, un mueble
de madera oscura que el padre del actual propietario había traído de un viaje
a Suiza en los años cuarenta y que seguía funcionando con sorprendente
precisión.

Una de las características que más sorprendía a los visitantes nuevos era el
sistema de organización. En lugar de separar los libros por géneros, como
hacía la mayoría de las librerías, allí se distribuían según una lógica
personal del dueño que combinaba autor, época y temas relacionados. Así, un
ensayo sobre filosofía de la ciencia podía estar junto a una novela del siglo
diecinueve si ambos tocaban la cuestión del progreso, o una colección de
poesía contemporánea acompañar a un manual de jardinería si el poeta era
aficionado a las plantas. Los clientes habituales aprendían a moverse con
soltura por estos criterios y muchos confesaban que descubrían títulos
inesperados precisamente gracias a esta forma de exponer.

Los miércoles por la mañana, cuando llegaban las cajas con los pedidos
nuevos, el librero dedicaba un par de horas a revisar cada ejemplar antes de
colocarlo en su sitio. A veces, cuando un libro le interesaba especialmente,
lo apartaba para llevárselo a casa y leerlo durante el fin de semana.
Después, si le había gustado, escribía a mano una pequeña ficha de
recomendación que pegaba con cinta adhesiva en la portada, una costumbre que
había heredado de su padre y que mantenía con la misma constancia que el
horario de apertura, las cuentas trimestrales y la limpieza del escaparate
todos los jueves al mediodía.
"""


# ── Modelo de errores OCR ────────────────────────────────────────────────────
# Confusiones visuales típicas de OCR. Cada letra apunta a una lista de
# posibles "alternativas" que el OCR podría producir por error.
OCR_CONFUSIONS = {
    'a': ['o', 'e', 'd'],
    'b': ['h', 'lo', 'd'],
    'c': ['e', 'o', 'r'],
    'd': ['a', 'cl', 'b'],
    'e': ['c', 'o', 'a'],
    'f': ['t', 'l'],
    'g': ['q', '8', '9'],
    'h': ['b', 'n', 'li'],
    'i': ['l', '1', 'j', 't'],
    'j': ['i', 'g'],
    'k': ['l'],
    'l': ['i', '1', 't'],
    'm': ['n', 'rn', 'in'],
    'n': ['m', 'h', 'u', 'r'],
    'o': ['c', 'a', '0', 'u'],
    'p': ['q', 'r'],
    'q': ['g', 'p', '9'],
    'r': ['n', 'i', 't'],
    's': ['5', '8'],
    't': ['l', 'f', 'i', 'r'],
    'u': ['v', 'n', 'ii'],
    'v': ['u', 'y', 'r'],
    'w': ['vv', 'm'],
    'x': ['k'],
    'y': ['v', 'g'],
    'z': ['s', '7'],
    'á': ['a', 'á'],
    'é': ['e', 'è'],
    'í': ['i', '1', 'l'],
    'ó': ['o', 'ò'],
    'ú': ['u', 'ù'],
    'ñ': ['n', 'fi', 'ii'],
}

# Misma tabla en versión mayúscula.
OCR_CONFUSIONS_UPPER = {k.upper(): [v.upper() for v in vs]
                        for k, vs in OCR_CONFUSIONS.items()}
OCR_CONFUSIONS_ALL = {**OCR_CONFUSIONS, **OCR_CONFUSIONS_UPPER}


def inject_ocr_errors(text: str, error_rate: float = 0.10,
                      rng: random.Random | None = None) -> str:
    """
    Inyecta errores OCR realistas. Para cada carácter de letra, hay
    probabilidad `error_rate` de mutarlo. Si se muta, con 70% se sustituye
    por una confusión visual, 15% se borra, 15% se duplica o se inserta.
    """
    if rng is None:
        rng = random.Random()
    out = []
    for ch in text:
        if not ch.isalpha():
            out.append(ch)
            continue
        if rng.random() > error_rate:
            out.append(ch)
            continue
        roll = rng.random()
        if roll < 0.70:
            # Sustitución por confusión visual
            confs = OCR_CONFUSIONS_ALL.get(ch)
            if confs:
                out.append(rng.choice(confs))
            else:
                out.append(ch)
        elif roll < 0.85:
            # Borrado: omitimos el carácter
            pass
        else:
            # Duplicado o inserción aleatoria
            if rng.random() < 0.5:
                out.append(ch * 2)
            else:
                out.append(ch + rng.choice("aeionrslc"))
    return "".join(out)


# ── Métricas ────────────────────────────────────────────────────────────────

def levenshtein(a, b) -> int:
    """Distancia de edición sobre cualquier secuencia indexable."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n]


_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_words(text: str) -> list:
    return _TOK_RE.findall(text)


def wer(reference: str, hypothesis: str) -> float:
    ref = tokenize_words(reference)
    hyp = tokenize_words(hypothesis)
    if not ref:
        return 0.0
    return levenshtein(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0
    return levenshtein(reference, hypothesis) / len(reference)


def count_word_changes(before: str, after: str) -> int:
    """Cuántas palabras cambian entre dos versiones (alineación posicional)."""
    b = tokenize_words(before)
    a = tokenize_words(after)
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


# ── Diff word-by-word para mostrar ejemplos ────────────────────────────────

def find_correction_examples(reference: str, ocr: str, corrected: str,
                             limit: int = 12) -> list:
    """
    Devuelve ejemplos donde el corrector cambió algo.
    Cada ejemplo: (referencia, ocr, corregido, tipo)
        tipo ∈ {'útil', 'dañina', 'sin-cambio'}
    """
    ref_w = tokenize_words(reference)
    ocr_w = tokenize_words(ocr)
    cor_w = tokenize_words(corrected)
    n = min(len(ref_w), len(ocr_w), len(cor_w))
    examples = []
    for i in range(n):
        if ocr_w[i] == cor_w[i]:
            continue  # corrector no tocó esta palabra
        if ref_w[i].lower() == cor_w[i].lower():
            tipo = "ÚTIL    "
        elif ref_w[i].lower() == ocr_w[i].lower():
            tipo = "DAÑINA  "
        else:
            tipo = "PARCIAL "
        examples.append((ref_w[i], ocr_w[i], cor_w[i], tipo))
        if len(examples) >= limit:
            break
    return examples


# ── Comando ─────────────────────────────────────────────────────────────────

class Command(BaseCommand):
    help = ("Compara SymSpell solo vs. SymSpell+BETO sobre texto español "
            "sintético con errores OCR realistas.")

    def add_arguments(self, parser):
        parser.add_argument(
            "--noise", type=float, default=0.10,
            help="Tasa de error a inyectar por carácter (0-1). Default 0.10.",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Semilla aleatoria (default 42).",
        )
        parser.add_argument(
            "--no-bert", action="store_true",
            help="Salta el modo BETO. Útil si transformers no está instalado.",
        )
        parser.add_argument(
            "--corpus", type=str, default=None,
            help="Ruta a un fichero .txt con el texto limpio. Si no se "
                 "indica, usa el texto embebido.",
        )
        parser.add_argument(
            "--examples", type=int, default=12,
            help="Cuántos ejemplos de corrección mostrar (default 12).",
        )

    def handle(self, *args, **opts):
        from apps.ocr import spell_correct  # late import: setup Django primero

        noise: float = opts["noise"]
        seed:  int   = opts["seed"]
        skip_bert    = opts["no_bert"]
        corpus_path  = opts["corpus"]
        n_examples   = opts["examples"]

        # 1. Texto limpio
        if corpus_path:
            text_clean = Path(corpus_path).read_text(encoding="utf-8")
        else:
            text_clean = DEFAULT_CORPUS

        n_words = len(tokenize_words(text_clean))
        n_chars = len(text_clean)

        # 2. Inyectar errores OCR
        rng = random.Random(seed)
        text_ocr = inject_ocr_errors(text_clean, error_rate=noise, rng=rng)

        # 3. Modos de corrección
        results = {}

        # 3a. Sin corrección (baseline = el OCR sucio)
        results["Sin corrección"] = {
            "text":  text_ocr,
            "wer":   wer(text_clean, text_ocr),
            "cer":   cer(text_clean, text_ocr),
            "changes": 0,
            "time":  0.0,
        }

        # 3b. SymSpell solo
        self.stdout.write("Cargando SymSpell (~3 s)...")
        t0 = time.perf_counter()
        text_sym = spell_correct.correct_text(text_ocr, use_bert=False)
        t_sym = time.perf_counter() - t0
        results["SymSpell"] = {
            "text":  text_sym,
            "wer":   wer(text_clean, text_sym),
            "cer":   cer(text_clean, text_sym),
            "changes": count_word_changes(text_ocr, text_sym),
            "time":  t_sym,
        }

        # 3c. SymSpell + BETO
        if not skip_bert:
            self.stdout.write(
                "Cargando BETO (~30-60 s la primera vez, luego en memoria)..."
            )
            t0 = time.perf_counter()
            text_bert = spell_correct.correct_text(text_ocr, use_bert=True)
            t_bert = time.perf_counter() - t0
            available = spell_correct.is_bert_available()
            label = "SymSpell + BETO" if available else "SymSpell + BETO (NO DISP.)"
            results[label] = {
                "text":  text_bert,
                "wer":   wer(text_clean, text_bert),
                "cer":   cer(text_clean, text_bert),
                "changes": count_word_changes(text_ocr, text_bert),
                "time":  t_bert,
            }

        # 4. Mostrar resultados
        bar = "─" * 76
        self.stdout.write("")
        self.stdout.write("═" * 76)
        self.stdout.write(" BENCHMARK DE CORRECCIÓN ORTOGRÁFICA")
        self.stdout.write("═" * 76)
        self.stdout.write(
            f" Corpus de referencia:  {n_words} palabras / {n_chars} caracteres"
        )
        self.stdout.write(
            f" Ruido OCR inyectado:   ~{noise*100:.0f}% de los caracteres alterados"
        )
        self.stdout.write(f" Semilla aleatoria:     {seed}")
        self.stdout.write("")
        self.stdout.write(bar)
        self.stdout.write(f" {'Modo':<28} │ {'WER':>7} │ {'CER':>7} │ {'Cambios':>8} │ {'Tiempo':>8}")
        self.stdout.write(bar)
        for label, r in results.items():
            wer_pct = f"{r['wer']*100:>5.1f}%"
            cer_pct = f"{r['cer']*100:>5.1f}%"
            changes = "—" if r['changes'] == 0 and "Sin" in label else str(r['changes'])
            t = "—" if r['time'] == 0 else f"{r['time']:>5.2f}s"
            self.stdout.write(
                f" {label:<28} │ {wer_pct:>7} │ {cer_pct:>7} │ {changes:>8} │ {t:>8}"
            )
        self.stdout.write(bar)

        # Mejoras relativas
        baseline = results["Sin corrección"]["wer"]
        sym_wer = results["SymSpell"]["wer"]
        self.stdout.write("")
        if baseline > 0:
            improvement_sym = (1 - sym_wer / baseline) * 100
            self.stdout.write(
                f" SymSpell reduce el WER un {improvement_sym:.0f}% sobre el OCR sucio."
            )
        if not skip_bert:
            bert_label = next(k for k in results if "BETO" in k)
            bert_wer = results[bert_label]["wer"]
            if baseline > 0:
                improvement_bert = (1 - bert_wer / baseline) * 100
                self.stdout.write(
                    f" SymSpell+BETO reduce el WER un {improvement_bert:.0f}% sobre el OCR sucio."
                )
            if sym_wer > 0:
                gain_over_sym = (1 - bert_wer / sym_wer) * 100
                if gain_over_sym > 0:
                    self.stdout.write(
                        f" BETO mejora un {gain_over_sym:.0f}% adicional sobre SymSpell solo."
                    )
                else:
                    self.stdout.write(
                        f" BETO NO mejora sobre SymSpell solo en este corpus "
                        f"({gain_over_sym:+.0f}%)."
                    )

        # 5. Ejemplos de corrección (modo BETO si está, si no SymSpell)
        if n_examples > 0:
            self.stdout.write("")
            self.stdout.write("─" * 76)
            best_label = next((k for k in results if "BETO" in k), "SymSpell")
            self.stdout.write(f" Ejemplos de cambios hechos por «{best_label}»:")
            self.stdout.write("─" * 76)
            self.stdout.write(
                f" {'TIPO':<9} │ {'REF':<14} │ {'OCR':<14} │ {'CORREGIDO':<14}"
            )
            self.stdout.write("─" * 76)
            examples = find_correction_examples(
                text_clean, text_ocr, results[best_label]["text"],
                limit=n_examples,
            )
            if not examples:
                self.stdout.write(" (sin ejemplos: no se cambió ninguna palabra)")
            else:
                for ref, ocr, cor, tipo in examples:
                    self.stdout.write(
                        f" {tipo:<9} │ {ref[:14]:<14} │ {ocr[:14]:<14} │ {cor[:14]:<14}"
                    )

        self.stdout.write("")
