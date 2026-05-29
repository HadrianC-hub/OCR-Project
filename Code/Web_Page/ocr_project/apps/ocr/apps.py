from django.apps import AppConfig


class OcrConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name  = 'apps.ocr'
    label = 'ocr'
    verbose_name = 'OCR'

    def ready(self):
        """
        Al arranque, realiza dos tareas:

        1. Recupera páginas que se quedaron en estado `processing` por
           un reinicio/crash del servidor y las resucita como `pending`.

        2. Pre-carga todos los modelos de IA en un hilo daemon para que
           estén listos antes de la primera solicitud OCR. La carga
           ocurre en segundo plano para no bloquear el arranque del
           servidor; los modelos usan sus locks internos de forma que
           el primer request que llegue antes de que termine la precarga
           simplemente espera a que el lock se libere (sin doble carga).

        Solo se ejecuta si el proceso es el worker real (no durante
        migraciones, tests u otros management commands).
        """
        import sys
        skip_commands = {'migrate', 'makemigrations', 'collectstatic',
                         'test', 'shell', 'dbshell', 'check'}
        if any(cmd in sys.argv for cmd in skip_commands):
            return

        try:
            from apps.ocr.tasks import recover_orphans
            recover_orphans()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("recover_orphans falló al arrancar.")

        # Pre-carga de modelos en hilo daemon.
        # Se importa threading aquí para no contaminar el namespace del módulo.
        import threading
        t = threading.Thread(target=_preload_models, daemon=True,
                             name="ocr-model-preload")
        t.start()


def _preload_models():
    """
    Invoca los getters de todos los modelos de IA para que sus singletons
    queden inicializados antes de la primera solicitud.

    Orden elegido para minimizar el tiempo de espera del primer request:
      1. kenLM (ligero, necesario por los predictores OCR)
      2. SymSpell (ligero, necesario para corrección)
      3. Predictor OCR impreso (modelo más frecuentemente usado)
      4. Predictor HTR manuscrito
      5. BETO ONNX (el más pesado; se carga al final)

    Cada getter ya incluye su propio lock y double-checked locking, de modo
    que si un request llega mientras este hilo está cargando, simplemente
    esperará al lock sin iniciar una segunda carga.
    """
    import logging
    log = logging.getLogger(__name__)

    steps = [
        ("kenLM",              _load_arpa_lm),
        ("SymSpell",           _load_symspell),
        ("OCR impreso",        _load_printed),
        ("HTR manuscrito",     _load_manuscript),
        ("BETO ONNX",          _load_bert),
    ]

    for name, loader in steps:
        try:
            result = loader()
            status = "OK" if result is not None else "no disponible (modelo ausente)"
            log.info("[preload] %-20s → %s", name, status)
        except Exception:
            log.exception("[preload] Error cargando %s", name)


# ── Loaders individuales (aíslan los imports para degradación elegante) ──

def _load_arpa_lm():
    from apps.ocr.ocr_engine import _get_arpa_lm
    return _get_arpa_lm()


def _load_printed():
    from apps.ocr.ocr_engine import _get_printed_predictor
    return _get_printed_predictor()


def _load_manuscript():
    from apps.ocr.ocr_engine import _get_manuscript_predictor
    return _get_manuscript_predictor()


def _load_symspell():
    from apps.ocr.spell_correct import _get_symspell
    return _get_symspell()


def _load_bert():
    from apps.ocr.spell_correct import _get_bert
    session, tokenizer, mask_id = _get_bert()
    return session  # None si no disponible, objeto si cargado
