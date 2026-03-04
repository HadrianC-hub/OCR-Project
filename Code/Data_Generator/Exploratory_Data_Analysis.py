import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from PIL import Image

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("Generated")
VOCAB_FILE  = Path("Vocabulary") / "vocab.txt"
EDA_DIR     = Path("EDA")
FIGURA_OUT  = EDA_DIR / "eda_dashboard.png"
INFORME_OUT = EDA_DIR / "eda_informe.txt"

# ── Paleta ────────────────────────────────────────────────────────────────────
C_AZUL    = "#2563EB"
C_VERDE   = "#16A34A"
C_ROJO    = "#DC2626"
C_NARANJA = "#D97706"
C_GRIS    = "#6B7280"
C_VIOLETA = "#7C3AED"
C_AMBAR   = "#F59E0B"
C_FONDO   = "#F9FAFB"

# ── Importar frecuencias objetivo desde el generador ──────────────────────────
# Se usa para la comparación objetivo vs. real (G6).
try:
    from generator import FRECUENCIAS_OBJETIVO
    _TIENE_FRECUENCIAS = True
except ImportError:
    FRECUENCIAS_OBJETIVO = {}
    _TIENE_FRECUENCIAS = False
    print("AVISO: No se pudo importar FRECUENCIAS_OBJETIVO desde generator.py")
    print("       La gráfica de comparación objetivo vs. real no estará disponible.")

STRIDE_CNN = 4

# ══════════════════════════════════════════════════════════════════════════════
#  1. CARGA DE MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_metricas(dataset_dir: Path) -> dict:
    """
    Recorre todos los pares .txt/.png del dataset y extrae:
      - Dimensiones de cada imagen (ancho, alto, aspect ratio)
      - Longitud de la etiqueta de texto
      - Fuente tipográfica de cada imagen
      - Recuento global de caracteres y palabras
      - Recuento de caracteres por fuente
      - Detección de imágenes degeneradas (sin contenido visible)
    """
    archivos_txt = sorted([
        f for f in dataset_dir.glob("*.txt")
        if f.stem != "ESTADISTICAS_DATASET"
    ])
    if not archivos_txt:
        raise FileNotFoundError(f"No se encontraron .txt en '{dataset_dir}'")

    anchos, altos, aspect_ratios = [], [], []
    longitudes, fuentes_lista    = [], []
    chars_global   = Counter()
    chars_x_fuente = {}
    palabras_todas = []
    errores_lectura    = 0
    imagenes_vacias    = 0

    for txt_path in archivos_txt:
        png_path    = txt_path.with_suffix(".png")
        partes      = txt_path.stem.split("_")
        nombre_font = "_".join(partes[:-1])
        texto       = txt_path.read_text(encoding="utf-8").strip()

        longitudes.append(len(texto))
        fuentes_lista.append(nombre_font)
        chars_global.update(texto)
        palabras_todas.extend(texto.split())

        if nombre_font not in chars_x_fuente:
            chars_x_fuente[nombre_font] = Counter()
        chars_x_fuente[nombre_font].update(texto)

        if png_path.exists():
            try:
                with Image.open(png_path) as img:
                    w_img, h_img = img.width, img.height
                    anchos.append(w_img)
                    altos.append(h_img)
                    aspect_ratios.append(w_img / h_img if h_img > 0 else 0)

                    # Detectar imágenes degeneradas: sin píxeles oscuros (sin texto)
                    arr = np.array(img.convert("L"))
                    if arr.min() > 200:   # toda la imagen es casi blanca
                        imagenes_vacias += 1
            except Exception:
                errores_lectura += 1
                anchos.append(0); altos.append(0); aspect_ratios.append(0)
        else:
            errores_lectura += 1
            anchos.append(0); altos.append(0); aspect_ratios.append(0)

    fuentes_unicas = sorted(set(fuentes_lista))
    conteo_fuente  = Counter(fuentes_lista)

    return {
        "n_total":          len(archivos_txt),
        "anchos":           np.array(anchos),
        "altos":            np.array(altos),
        "aspect_ratios":    np.array(aspect_ratios),
        "longitudes":       np.array(longitudes),
        "fuentes_lista":    fuentes_lista,
        "fuentes_unicas":   fuentes_unicas,
        "conteo_fuente":    conteo_fuente,
        "chars_global":     chars_global,
        "chars_x_fuente":   chars_x_fuente,
        "palabras_todas":   palabras_todas,
        "errores_lectura":  errores_lectura,
        "imagenes_vacias":  imagenes_vacias,
        "archivos_txt":     archivos_txt,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  2. ALERTAS AUTOMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════

def analizar_alertas(m: dict) -> list:
    alertas = []

    # ── Integridad del dataset ────────────────────────────────────────────────
    if m["errores_lectura"] > 0:
        alertas.append(("ERROR",
            f"{m['errores_lectura']} imágenes no pudieron leerse. "
            f"Verificar integridad del dataset."))
    else:
        alertas.append(("OK", "Todas las imágenes se leyeron correctamente."))

    if m["imagenes_vacias"] > 0:
        pct = m["imagenes_vacias"] / m["n_total"] * 100
        alertas.append(("ERROR",
            f"{m['imagenes_vacias']} imágenes ({pct:.1f}%) están completamente "
            f"blancas (sin texto visible). Posible fallo silencioso del renderizador."))
    else:
        alertas.append(("OK", "Ninguna imagen degenerada detectada (todas tienen contenido visible)."))

    # ── Restricción CTC ───────────────────────────────────────────────────────
    min_ancho_requerido = m["longitudes"] * STRIDE_CNN
    violaciones = int(np.sum(m["anchos"] < min_ancho_requerido))
    pct_ctc = violaciones / m["n_total"] * 100
    if violaciones > 0:
        alertas.append((
            "ERROR" if pct_ctc > 5 else "AVISO",
            f"CTC: {violaciones:,} imágenes ({pct_ctc:.1f}%) violan restricción "
            f"ancho >= longitud×{STRIDE_CNN}. Solución: reducir MAX_CHARS_PER_LINE "
            f"o aumentar MARGIN_LEFT/RIGHT."))
    else:
        alertas.append(("OK",
            f"CTC: todas las imágenes cumplen restricción ancho >= longitud×{STRIDE_CNN}."))

    # ── Variabilidad de ancho (eficiencia de padding en batches) ──────────────
    av = m["anchos"][m["anchos"] > 0]
    if len(av) > 0:
        cv_ancho = av.std() / av.mean() * 100
        if cv_ancho > 80:
            alertas.append(("AVISO",
                f"Anchos muy variables (CV={cv_ancho:.0f}%). Padding ineficiente "
                f"en batches. Considerar bucketing por longitud de secuencia."))
        else:
            alertas.append(("OK", f"Variabilidad de anchos aceptable (CV={cv_ancho:.0f}%)."))

    # ── Balance entre fuentes ─────────────────────────────────────────────────
    conteos = list(m["conteo_fuente"].values())
    if conteos:
        ratio = max(conteos) / max(min(conteos), 1)
        if ratio > 1.15:
            alertas.append(("AVISO",
                f"Desbalance entre fuentes: ratio max/min = {ratio:.2f}. "
                f"Puede sesgar el modelo hacia las fuentes más representadas."))
        else:
            alertas.append(("OK",
                f"Fuentes bien balanceadas (ratio max/min = {ratio:.2f})."))

    # ── Longitudes extremas ───────────────────────────────────────────────────
    muy_cortas = int(np.sum(m["longitudes"] < 5))
    muy_largas = int(np.sum(m["longitudes"] > 90))
    if muy_cortas > 0:
        alertas.append(("AVISO",
            f"{muy_cortas} etiquetas con <5 chars. Poco contexto para BiLSTM."))
    if muy_largas > 0:
        alertas.append(("AVISO",
            f"{muy_largas} etiquetas con >90 chars. Verificar compatibilidad con arquitectura."))
    if muy_cortas == 0 and muy_largas == 0:
        alertas.append(("OK", "Todas las longitudes de etiqueta están en rango óptimo (5–90 chars)."))

    # ── Cruce con vocab.txt ───────────────────────────────────────────────────
    if VOCAB_FILE.exists():
        vocab_chars = set(
            line.strip() for line in VOCAB_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        fuera_vocab = {
            c: n for c, n in m["chars_global"].items()
            if c not in vocab_chars and c.strip() and c.isprintable()
        }
        sin_ejemplos = [c for c in vocab_chars if c not in m["chars_global"] and c.strip()]

        if fuera_vocab:
            chars_fuera = "".join(fuera_vocab.keys())
            alertas.append(("ERROR",
                f"Chars en dataset NO cubiertos por vocab.txt: '{chars_fuera}'. "
                f"El modelo no podrá predecirlos. Actualizar vocab.txt."))
        else:
            alertas.append(("OK", "Todos los chars del dataset están cubiertos por vocab.txt."))

        if sin_ejemplos:
            alertas.append(("AVISO",
                f"{len(sin_ejemplos)} chars de vocab.txt sin ejemplos en el dataset: "
                f"'{''.join(sin_ejemplos)}'. El modelo los tendrá en su alfabeto "
                f"pero sin entrenamiento real."))
    else:
        alertas.append(("AVISO",
            f"No se encontró '{VOCAB_FILE}'. Ejecutar generar_vocab.py primero."))

    # ── Cobertura mínima de chars especiales ──────────────────────────────────
    chars_criticos = list("ñÑüÜáéíóúÁÉÍÓÚ¿¡—«»")
    insuficientes = [c for c in chars_criticos
                     if m["chars_global"].get(c, 0) < 500]
    if insuficientes:
        alertas.append(("AVISO",
            f"Chars especiales con <500 apariciones: '{''.join(insuficientes)}'. "
            f"Cobertura insuficiente para entrenamiento confiable."))
    else:
        alertas.append(("OK",
            "Todos los chars especiales del español tienen ≥500 apariciones."))

    return alertas

# ══════════════════════════════════════════════════════════════════════════════
#  3. FIGURA PARA LA TESIS
# ══════════════════════════════════════════════════════════════════════════════

def generar_figura(m: dict, alertas: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Grid: 5 filas × 3 columnas
    # Fila 0: G1 fuentes | G2 longitudes | G3 anchos
    # Fila 1: G4 top30 chars (2 cols) | G5 chars especiales
    # Fila 2: G6 objetivo vs real (2 cols) | G7 CTC scatter
    # Fila 3: G8 alturas por fuente | G9 aspect ratio | G10 palabras/vocab stats
    # Fila 4: G11 panel de alertas (full row)

    fig = plt.figure(figsize=(21, 28), facecolor=C_FONDO)
    fig.suptitle(
        "Análisis Exploratorio del Dataset Sintético OCR — Español",
        fontsize=18, fontweight="bold", y=0.99, color="#111827"
    )

    gs = gridspec.GridSpec(5, 3, figure=fig,
                           hspace=0.52, wspace=0.38,
                           top=0.96, bottom=0.03, left=0.07, right=0.97)

    paleta       = plt.cm.tab10.colors
    color_fuente = {f: paleta[i % len(paleta)] for i, f in enumerate(m["fuentes_unicas"])}

    def abrev(nombre):
        p = nombre.split("-")[0]
        return p[:13] if len(p) > 13 else p

    fuentes_abrev = {f: abrev(f) for f in m["fuentes_unicas"]}

    # ── G1: Imágenes por fuente ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    nombres = [fuentes_abrev[f] for f in m["fuentes_unicas"]]
    valores = [m["conteo_fuente"][f] for f in m["fuentes_unicas"]]
    colores = [color_fuente[f] for f in m["fuentes_unicas"]]
    bars = ax1.barh(nombres, valores, color=colores, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, valores):
        ax1.text(bar.get_width() + max(valores) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{v:,}", va="center", fontsize=8, color=C_GRIS)
    ax1.set_title("Imágenes por fuente tipográfica", fontsize=11, fontweight="bold")
    ax1.set_xlabel("N° imágenes")
    ax1.tick_params(axis="y", labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_facecolor(C_FONDO)

    # ── G2: Distribución de longitudes de etiqueta ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(m["longitudes"], bins=40, color=C_AZUL,
             edgecolor="white", linewidth=0.3, alpha=0.85)
    med = np.median(m["longitudes"])
    mu  = np.mean(m["longitudes"])
    ax2.axvline(med, color=C_ROJO,  linestyle="--", linewidth=1.5, label=f"Mediana: {med:.0f}")
    ax2.axvline(mu,  color=C_VERDE, linestyle=":",  linewidth=1.5, label=f"Media: {mu:.1f}")
    ax2.set_title("Longitud de etiquetas (chars/imagen)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Caracteres"); ax2.set_ylabel("Frecuencia")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor(C_FONDO)

    # ── G3: Distribución de anchos de imagen ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    av = m["anchos"][m["anchos"] > 0]
    ax3.hist(av, bins=40, color=C_VERDE, edgecolor="white", linewidth=0.3, alpha=0.85)
    ax3.axvline(np.median(av), color=C_ROJO, linestyle="--", linewidth=1.5,
                label=f"Mediana: {np.median(av):.0f}px")
    ax3.set_title("Anchura de imágenes (píxeles)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Píxeles"); ax3.set_ylabel("Frecuencia")
    ax3.legend(fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor(C_FONDO)

    # ── G4: Top 30 caracteres por frecuencia ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    top30     = m["chars_global"].most_common(30)
    chars_top = [c if c != " " else "⎵" for c, _ in top30]
    total_ch  = sum(m["chars_global"].values())
    pcts      = [n / total_ch * 100 for _, n in top30]
    bar_cols  = []
    for c, _ in top30:
        if c.isalpha() and c.islower():   bar_cols.append(C_AZUL)
        elif c.isalpha() and c.isupper(): bar_cols.append(C_VIOLETA)
        elif c.isdigit():                 bar_cols.append(C_VERDE)
        else:                             bar_cols.append(C_AMBAR)
    ax4.bar(chars_top, pcts, color=bar_cols, edgecolor="white", linewidth=0.3)
    ax4.set_title("Frecuencia relativa — top 30 caracteres",
                  fontsize=11, fontweight="bold")
    ax4.set_ylabel("% sobre total")
    ax4.tick_params(axis="x", labelsize=9)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.set_facecolor(C_FONDO)
    ax4.legend(handles=[
        Patch(color=C_AZUL,    label="Minúscula"),
        Patch(color=C_VIOLETA, label="Mayúscula"),
        Patch(color=C_VERDE,   label="Dígito"),
        Patch(color=C_AMBAR,   label="Puntuación"),
    ], fontsize=8, loc="upper right")

    # ── G5: Cobertura de caracteres especiales del español ────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    chars_esp = list("ñÑáéíóúÁÉÍÓÚüÜ¿¡—«»")
    cnts_esp  = [m["chars_global"].get(c, 0) for c in chars_esp]
    cols5 = [C_VERDE if n >= 500 else (C_ROJO if n == 0 else C_AMBAR) for n in cnts_esp]
    ax5.barh(chars_esp, cnts_esp, color=cols5, edgecolor="white", linewidth=0.3)
    ax5.axvline(500, color=C_ROJO, linestyle="--", linewidth=1, label="Mín. recomendado (500)")
    ax5.set_title("Cobertura de chars\nespeciales del español", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Apariciones")
    ax5.tick_params(axis="y", labelsize=9)
    ax5.legend(fontsize=7)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.set_facecolor(C_FONDO)

    # ── G6: Frecuencia objetivo vs. real ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    if _TIENE_FRECUENCIAS and FRECUENCIAS_OBJETIVO:
        total_peso = sum(FRECUENCIAS_OBJETIVO.values())
        # Mostrar solo los 40 caracteres con mayor frecuencia objetivo
        chars_sorted = sorted(FRECUENCIAS_OBJETIVO.keys(),
                              key=lambda c: FRECUENCIAS_OBJETIVO[c], reverse=True)[:40]

        obj_pcts  = [FRECUENCIAS_OBJETIVO[c] / total_peso * 100 for c in chars_sorted]
        real_pcts = [m["chars_global"].get(c, 0) / max(total_ch, 1) * 100 for c in chars_sorted]

        x  = np.arange(len(chars_sorted))
        bw = 0.38
        ax6.bar(x - bw/2, obj_pcts,  width=bw, color=C_AZUL,   alpha=0.85,
                label="Frecuencia objetivo", edgecolor="white", linewidth=0.2)
        ax6.bar(x + bw/2, real_pcts, width=bw, color=C_NARANJA, alpha=0.85,
                label="Frecuencia real",     edgecolor="white", linewidth=0.2)
        ax6.set_xticks(x)
        ax6.set_xticklabels(
            [c if c not in (" ",) else "⎵" for c in chars_sorted],
            fontsize=7.5
        )
        ax6.set_title("Distribución de caracteres: frecuencia objetivo vs. real obtenida",
                      fontsize=11, fontweight="bold")
        ax6.set_ylabel("% sobre total de caracteres")
        ax6.legend(fontsize=9)
        ax6.spines[["top", "right"]].set_visible(False)
        ax6.set_facecolor(C_FONDO)
    else:
        ax6.text(0.5, 0.5, "FRECUENCIAS_OBJETIVO no disponible\n(importar desde generator.py)",
                 ha="center", va="center", transform=ax6.transAxes,
                 fontsize=11, color=C_GRIS)
        ax6.axis("off")

    # ── G7: Scatter ancho vs. longitud (restricción CTC) ─────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    n_sample = min(3000, len(m["anchos"]))
    idx_s    = np.random.choice(len(m["anchos"]), size=n_sample, replace=False)
    anch_s   = m["anchos"][idx_s]
    long_s   = m["longitudes"][idx_s]
    ok_mask  = anch_s >= long_s * STRIDE_CNN
    ax7.scatter(long_s[ok_mask],  anch_s[ok_mask],
                alpha=0.15, s=3, color=C_AZUL, label="OK")
    ax7.scatter(long_s[~ok_mask], anch_s[~ok_mask],
                alpha=0.6,  s=8, color=C_ROJO, label="Viola CTC")
    xs = np.linspace(0, max(long_s.max(), 5), 100)
    ax7.plot(xs, xs * STRIDE_CNN, color=C_ROJO, linestyle="--",
             linewidth=1.5, label=f"Límite CTC (×{STRIDE_CNN})")
    ax7.set_title("Ancho imagen vs. longitud\netiqueta — restricción CTC",
                  fontsize=11, fontweight="bold")
    ax7.set_xlabel("Chars en etiqueta")
    ax7.set_ylabel("Ancho imagen (px)")
    ax7.legend(fontsize=7)
    ax7.spines[["top", "right"]].set_visible(False)
    ax7.set_facecolor(C_FONDO)

    # ── G8: Distribución de alturas por fuente ────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 0])
    datos_alt = []
    etiq_alt  = []
    for f in m["fuentes_unicas"]:
        alts_f = [a for a, fn in zip(m["altos"], m["fuentes_lista"])
                  if fn == f and a > 0]
        if alts_f:
            datos_alt.append(alts_f)
            etiq_alt.append(fuentes_abrev[f])
    if datos_alt:
        bp = ax8.boxplot(datos_alt, labels=etiq_alt, patch_artist=True,
                         medianprops=dict(color=C_ROJO, linewidth=2))
        for patch, f in zip(bp["boxes"], m["fuentes_unicas"]):
            patch.set_facecolor(color_fuente[f])
            patch.set_alpha(0.7)
    ax8.set_title("Altura de imagen por fuente\n(px — afecta escala al normalizar)",
                  fontsize=10, fontweight="bold")
    ax8.set_ylabel("Altura (px)")
    ax8.tick_params(axis="x", labelsize=7, rotation=15)
    ax8.spines[["top", "right"]].set_visible(False)
    ax8.set_facecolor(C_FONDO)

    # ── G9: Distribución de aspect ratio ─────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 1])
    ar_validos = m["aspect_ratios"][m["aspect_ratios"] > 0]
    ax9.hist(ar_validos, bins=40, color=C_VIOLETA,
             edgecolor="white", linewidth=0.3, alpha=0.85)
    ax9.axvline(np.median(ar_validos), color=C_ROJO, linestyle="--", linewidth=1.5,
                label=f"Mediana: {np.median(ar_validos):.1f}")
    p95 = np.percentile(ar_validos, 95)
    ax9.axvline(p95, color=C_NARANJA, linestyle=":", linewidth=1.5,
                label=f"P95: {p95:.1f}")
    ax9.set_title("Distribución de aspect ratio\n(ancho/alto — impacta padding en batches)",
                  fontsize=10, fontweight="bold")
    ax9.set_xlabel("Ancho / Alto")
    ax9.set_ylabel("Frecuencia")
    ax9.legend(fontsize=8)
    ax9.spines[["top", "right"]].set_visible(False)
    ax9.set_facecolor(C_FONDO)

    # ── G10: Estadísticas de palabras y cobertura de vocabulario ──────────────
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis("off")
    ax10.set_facecolor(C_FONDO)
    ax10.set_title("Estadísticas de corpus\ny vocabulario", fontsize=10, fontweight="bold")

    n_palabras  = len(m["palabras_todas"])
    n_unicas    = len(set(m["palabras_todas"]))
    n_chars_tok = sum(m["chars_global"].values())
    n_chars_uni = len(m["chars_global"])

    # CV de frecuencias de caracteres
    freqs = list(m["chars_global"].values())
    media_f = sum(freqs) / len(freqs) if freqs else 1
    cv_f = (sum((x - media_f)**2 for x in freqs) / len(freqs))**0.5 / media_f * 100

    # Cobertura vocab
    vocab_cov = "N/A"
    if VOCAB_FILE.exists():
        vocab_chars = set(
            l.strip() for l in VOCAB_FILE.read_text(encoding="utf-8").splitlines() if l.strip()
        )
        chars_en_dataset = {c for c in m["chars_global"] if c.strip()}
        cubiertos = chars_en_dataset & vocab_chars
        vocab_cov = f"{len(cubiertos)}/{len(vocab_chars)} ({len(cubiertos)/max(len(vocab_chars),1)*100:.0f}%)"

    lineas_info = [
        ("Total imágenes",    f"{m['n_total']:,}"),
        ("Total palabras",    f"{n_palabras:,}"),
        ("Palabras únicas",   f"{n_unicas:,}"),
        ("Total chars",       f"{n_chars_tok:,}"),
        ("Chars únicos",      f"{n_chars_uni}"),
        ("Vocab cubierto",    vocab_cov),
        ("CV frecuencias",    f"{cv_f:.1f}%"),
        ("Imgs degeneradas",  f"{m['imagenes_vacias']}"),
        ("Errores lectura",   f"{m['errores_lectura']}"),
    ]
    y0 = 0.95
    for etiq, val in lineas_info:
        ax10.text(0.05, y0, etiq, transform=ax10.transAxes,
                  fontsize=9, color=C_GRIS, va="top")
        ax10.text(0.95, y0, val, transform=ax10.transAxes,
                  fontsize=9, color="#111827", va="top", ha="right", fontweight="bold")
        y0 -= 0.10

    # ── G11: Panel de alertas ─────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[4, :])
    ax11.axis("off")
    ax11.set_facecolor(C_FONDO)
    ax11.set_title("Diagnóstico automático del dataset",
                   fontsize=11, fontweight="bold", loc="left", pad=8)

    iconos  = {"ERROR": "✗", "AVISO": "⚠", "OK": "✓"}
    col_niv = {"ERROR": C_ROJO, "AVISO": C_NARANJA, "OK": C_VERDE}

    # Distribuir alertas en dos columnas
    n = len(alertas)
    mitad = (n + 1) // 2
    for col, subset in enumerate([alertas[:mitad], alertas[mitad:]]):
        y_pos = 0.94
        for nivel, msg in subset:
            icono  = iconos[nivel]
            color  = col_niv[nivel]
            texto  = f"{icono} [{nivel}]  {msg}"
            # Truncar si es muy largo
            if len(texto) > 90:
                texto = texto[:87] + "..."
            ax11.text(0.01 + col * 0.50, y_pos, texto,
                      transform=ax11.transAxes,
                      fontsize=8.5, color=color, va="top", family="monospace")
            y_pos -= 0.115
            if y_pos < 0.01:
                break

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=C_FONDO, edgecolor="none")
    plt.close()
    print(f"Figura guardada -> {output_path.resolve()}")

# ══════════════════════════════════════════════════════════════════════════════
#  4. INFORME DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

def generar_informe(m: dict, alertas: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    av    = m["anchos"][m["anchos"] > 0]
    altv  = m["altos"][m["altos"] > 0]
    arv   = m["aspect_ratios"][m["aspect_ratios"] > 0]
    total = sum(m["chars_global"].values())

    freqs  = list(m["chars_global"].values())
    media_f = sum(freqs) / len(freqs) if freqs else 1
    cv_f   = (sum((x - media_f)**2 for x in freqs) / len(freqs))**0.5 / media_f * 100

    with output_path.open("w", encoding="utf-8") as f:
        def w(s=""): f.write(s + "\n")

        w("=" * 72)
        w("INFORME EDA — DATASET SINTÉTICO OCR — ESPAÑOL")
        w("=" * 72); w()

        # ── 1. Resumen general ────────────────────────────────────────────────
        w("1. RESUMEN GENERAL")
        w("-" * 72)
        w(f"   Total imágenes          : {m['n_total']:>10,}")
        w(f"   Total fuentes           : {len(m['fuentes_unicas']):>10}")
        w(f"   Imágenes con error      : {m['errores_lectura']:>10,}")
        w(f"   Imágenes degeneradas    : {m['imagenes_vacias']:>10,}")
        w(f"   Total palabras          : {len(m['palabras_todas']):>10,}")
        w(f"   Palabras únicas         : {len(set(m['palabras_todas'])):>10,}")
        w(f"   Total caracteres        : {total:>10,}")
        w(f"   Chars únicos en dataset : {len(m['chars_global']):>10}")
        w(f"   CV frecuencias de chars : {cv_f:>9.1f}%")
        w()
        w("   Nota sobre el CV de frecuencias:")
        w("   Un CV elevado es esperable en lenguaje natural (ley de Zipf).")
        w("   La métrica relevante para OCR es la cobertura mínima por carácter")
        w("   (sección 5), no el CV global.")
        w()

        # ── 2. Dimensiones de imagen ──────────────────────────────────────────
        w("2. DIMENSIONES DE IMAGEN")
        w("-" * 72)
        if len(av) > 0:
            w(f"   Ancho — media        : {av.mean():>8.1f} px")
            w(f"   Ancho — mediana      : {np.median(av):>8.1f} px")
            w(f"   Ancho — min / max    : {av.min():>5} / {av.max()} px")
            w(f"   Ancho — std (CV)     : {av.std():>8.1f} px  (CV={av.std()/av.mean()*100:.1f}%)")
        if len(altv) > 0:
            w(f"   Alto  — media        : {altv.mean():>8.1f} px")
            w(f"   Alto  — mediana      : {np.median(altv):>8.1f} px")
            w(f"   Alto  — min / max    : {altv.min():>5} / {altv.max()} px")
        if len(arv) > 0:
            w(f"   Aspect ratio — media : {arv.mean():>8.2f}")
            w(f"   Aspect ratio — P50   : {np.median(arv):>8.2f}")
            w(f"   Aspect ratio — P95   : {np.percentile(arv, 95):>8.2f}")
            w(f"   Aspect ratio — max   : {arv.max():>8.2f}")
        w()
        w("   Alturas por fuente:")
        for fuente in m["fuentes_unicas"]:
            alts_f = [a for a, fn in zip(m["altos"], m["fuentes_lista"]) if fn == fuente and a > 0]
            if alts_f:
                w(f"   {fuente:<45} media={np.mean(alts_f):.1f}px  "
                  f"rango=[{min(alts_f)},{max(alts_f)}]")
        w()

        # ── 3. Longitudes de etiqueta ─────────────────────────────────────────
        w("3. LONGITUDES DE ETIQUETA")
        w("-" * 72)
        lo = m["longitudes"]
        w(f"   Media        : {lo.mean():.1f} chars")
        w(f"   Mediana      : {np.median(lo):.1f} chars")
        w(f"   Min / Max    : {lo.min()} / {lo.max()} chars")
        w(f"   Std          : {lo.std():.1f} chars")
        perc = np.percentile(lo, [5, 25, 75, 95])
        w(f"   P5/P25/P75/P95: {perc[0]:.0f} / {perc[1]:.0f} / {perc[2]:.0f} / {perc[3]:.0f}")
        w()
        w(f"   Restricción CTC (stride={STRIDE_CNN}):")
        w(f"   Ancho mínimo requerido para la etiqueta más larga "
          f"({lo.max()} chars): {lo.max() * STRIDE_CNN} px")
        violaciones = int(np.sum(m["anchos"] < lo * STRIDE_CNN))
        w(f"   Imágenes que violan la restricción: {violaciones:,} de {m['n_total']:,}")
        w()

        # ── 4. Distribución por fuente ────────────────────────────────────────
        w("4. DISTRIBUCIÓN POR FUENTE")
        w("-" * 72)
        w(f"   {'FUENTE':<45}  {'IMGS':>6}  {'%':>6}  {'L.MEDIA':>8}  {'L.STD':>7}")
        w(f"   {'─'*45}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*7}")
        for font in m["fuentes_unicas"]:
            cnt = m["conteo_fuente"][font]
            pct = cnt / m["n_total"] * 100
            lf  = [l for l, fn in zip(m["longitudes"], m["fuentes_lista"]) if fn == font]
            w(f"   {font:<45}  {cnt:>6,}  {pct:>5.1f}%  "
              f"{np.mean(lf):>7.1f}  {np.std(lf):>6.1f}")
        w()

        # ── 5. Cobertura de caracteres ────────────────────────────────────────
        w("5. COBERTURA DE CARACTERES")
        w("-" * 72)
        w(f"   {'CHAR':<8} {'OBJETIVO%':>10}  {'REAL%':>7}  {'DESV.':>7}  "
          f"{'APARIC.':>10}  ESTADO")
        w(f"   {'─'*8} {'─'*10}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*12}")

        total_peso = sum(FRECUENCIAS_OBJETIVO.values()) if FRECUENCIAS_OBJETIVO else 1
        chars_eval = sorted(m["chars_global"].keys(),
                            key=lambda c: m["chars_global"][c], reverse=True)

        for c in chars_eval:
            if not c.strip() or not c.isprintable():
                continue
            cnt      = m["chars_global"][c]
            real_pct = cnt / total * 100
            obj_pct  = FRECUENCIAS_OBJETIVO.get(c, 0) / total_peso * 100 if FRECUENCIAS_OBJETIVO else 0
            desv     = real_pct - obj_pct
            desv_str = f"{desv:+.2f}%" if obj_pct > 0 else "  N/A "

            if cnt >= 30000:   estado = "ABUNDANTE"
            elif cnt >= 500:   estado = "OK"
            elif cnt >= 100:   estado = "BAJO"
            else:              estado = "INSUFICIENTE"

            display = repr(c) if c == " " else f"'{c}'"
            obj_str = f"{obj_pct:.2f}%" if obj_pct > 0 else "    —  "
            w(f"   {display:<8} {obj_str:>10}  {real_pct:>6.2f}%  {desv_str:>7}  "
              f"{cnt:>10,}  {estado}")
        w()

        # ── 6. Cruce con vocabulario ──────────────────────────────────────────
        w("6. CRUCE CON VOCABULARIO (vocab.txt)")
        w("-" * 72)
        if VOCAB_FILE.exists():
            vocab_chars = set(
                l.strip() for l in VOCAB_FILE.read_text(encoding="utf-8").splitlines()
                if l.strip()
            )
            chars_dataset = {c for c in m["chars_global"] if c.strip() and c.isprintable()}
            cubiertos     = chars_dataset & vocab_chars
            fuera_vocab   = chars_dataset - vocab_chars
            sin_ejemplos  = vocab_chars - chars_dataset - {" "}

            w(f"   Chars en vocab.txt      : {len(vocab_chars)}")
            w(f"   Chars en dataset        : {len(chars_dataset)}")
            w(f"   Chars cubiertos         : {len(cubiertos)}")
            w(f"   Chars fuera del vocab   : {len(fuera_vocab)}"
              + (f"  → {''.join(sorted(fuera_vocab))}" if fuera_vocab else ""))
            w(f"   Chars sin ejemplos      : {len(sin_ejemplos)}"
              + (f"  → {''.join(sorted(sin_ejemplos))}" if sin_ejemplos else ""))
        else:
            w(f"   vocab.txt no encontrado en '{VOCAB_FILE}'")
            w("   Ejecutar generar_vocab.py antes del EDA para este análisis.")
        w()

        # ── 7. Diagnóstico automático ─────────────────────────────────────────
        w("7. DIAGNÓSTICO AUTOMÁTICO")
        w("-" * 72)
        for nivel, msg in alertas:
            icono = {"ERROR": "[ERROR]", "AVISO": "[AVISO]", "OK": "[ OK  ]"}[nivel]
            palabras = f"{icono} {msg}".split()
            linea    = "   "
            for p in palabras:
                if len(linea) + len(p) > 72:
                    w(linea)
                    linea = "   " + " " * 9 + p + " "
                else:
                    linea += p + " "
            w(linea.rstrip())
        w()
        w("=" * 72)

    print(f"Informe guardado -> {output_path.resolve()}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EDA — DATASET SINTÉTICO OCR — ESPAÑOL")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"ERROR: No se encontró la carpeta '{DATASET_DIR}'")
        exit(1)

    print(f"\nCargando métricas de '{DATASET_DIR}/'...")
    m = cargar_metricas(DATASET_DIR)
    print(f"  {m['n_total']:,} imgs  |  {len(m['fuentes_unicas'])} fuentes  |  "
          f"{len(m['chars_global'])} chars únicos  |  "
          f"{len(set(m['palabras_todas'])):,} palabras únicas")
    if m["imagenes_vacias"] > 0:
        print(f"  AVISO: {m['imagenes_vacias']} imágenes degeneradas detectadas")
    if m["errores_lectura"] > 0:
        print(f"  ERROR: {m['errores_lectura']} imágenes no pudieron leerse")

    print("\nAnalizando alertas...")
    alertas = analizar_alertas(m)
    for nivel, msg in alertas:
        prefijo = {"ERROR": "  [ERROR]", "AVISO": "  [AVISO]", "OK": "  [ OK  ]"}[nivel]
        print(f"{prefijo} {msg[:78]}{'...' if len(msg) > 78 else ''}")

    print("\nGenerando figura...")
    generar_figura(m, alertas, FIGURA_OUT)

    print("Generando informe...")
    generar_informe(m, alertas, INFORME_OUT)

    print("\n" + "=" * 60)
    print("EDA COMPLETADO")
    print(f"  Dashboard : {FIGURA_OUT}")
    print(f"  Informe   : {INFORME_OUT}")
    print("=" * 60)