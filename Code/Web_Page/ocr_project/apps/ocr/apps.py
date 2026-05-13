from django.apps import AppConfig


class OcrConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name  = 'apps.ocr'
    label = 'ocr'
    verbose_name = 'OCR'

    def ready(self):
        """
        Al arranque, busca páginas que se quedaron en estado `processing`
        por un reinicio/crash del servidor y las resucita como `pending`.
        El próximo POST a ocr_process del documento las re-procesará.

        Sólo se ejecuta una vez por proceso (Django llama ready() una vez
        por worker). Si runserver autoreloader está activo, ten en cuenta
        que ready() se llama también después de cada autoreload.
        """
        # Evitamos correr esto durante migraciones, makemigrations, etc.,
        # donde la tabla puede no existir todavía.
        import sys
        skip_commands = {'migrate', 'makemigrations', 'collectstatic',
                         'test', 'shell', 'dbshell', 'check'}
        if any(cmd in sys.argv for cmd in skip_commands):
            return

        try:
            from apps.ocr.tasks import recover_orphans
            recover_orphans()
        except Exception:
            # No bloqueamos el arranque por esto
            import logging
            logging.getLogger(__name__).exception("recover_orphans falló al arrancar.")
