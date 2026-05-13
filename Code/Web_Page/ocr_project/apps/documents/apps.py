from django.apps import AppConfig


class DocumentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name  = 'apps.documents'
    label = 'documents'
    verbose_name = 'Documentos'

    def ready(self):
        # Side-effect import: registra los handlers de post_delete.
        from . import signals  # noqa: F401
