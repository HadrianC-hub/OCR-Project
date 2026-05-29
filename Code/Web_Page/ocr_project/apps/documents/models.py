from django.db import models
from django.conf import settings


# ── Image upload path ──────────────────────────────────────────────────────

def page_image_path(instance, filename):
    """Store facsimiles under media/facsimiles/<doc_id>/<filename>"""
    return f'facsimiles/{instance.document_id}/{filename}'


# ── Document ───────────────────────────────────────────────────────────────

class Document(models.Model):
    PRINTED    = 'printed'
    MANUSCRIPT = 'manuscript'
    TYPE_CHOICES = [
        (PRINTED,    'Impreso'),
        (MANUSCRIPT, 'Manuscrito'),
    ]

    title       = models.CharField(max_length=500, verbose_name='Título')
    year        = models.IntegerField(null=True, blank=True, verbose_name='Año')
    author      = models.CharField(max_length=300, blank=True, verbose_name='Autor')
    description = models.TextField(blank=True, verbose_name='Descripción')
    document_type = models.CharField(
        max_length=20, choices=TYPE_CHOICES, default=PRINTED, verbose_name='Tipo')
    is_public   = models.BooleanField(default=True, verbose_name='Público')
    use_bert_correction = models.BooleanField(
        default=False,
        verbose_name='Corrección con BERT',
        help_text='Usa el reranker neuronal BETO para escoger entre los '
                  'candidatos de SymSpell. Mucho más preciso en casos '
                  'ambiguos pero 100x más lento (1-5 s por página). '
                  'Recomendado para documentos donde la calidad importe '
                  'más que la velocidad.',
    )

    created_by  = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, related_name='created_documents', verbose_name='Creado por')
    last_modified_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='modified_documents',
        verbose_name='Última modificación por')

    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    total_views     = models.PositiveIntegerField(default=0)
    total_downloads = models.PositiveIntegerField(default=0)
    total_edits     = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.title

    def can_access(self, user):
        """Return True if the given user may read this document."""
        if self.is_public:
            return True
        return (user and user.is_authenticated and user.is_worker_or_above)

    class Meta:
        verbose_name          = 'Documento'
        verbose_name_plural   = 'Documentos'
        ordering              = ['-created_at']


# ── Page ───────────────────────────────────────────────────────────────────

class Page(models.Model):
    """
    Una página facsimilar de un documento.

    El texto transcrito se guarda en {MEDIA_ROOT}/transcripts/{document_id}/page_{order:03d}.xml
    y se accede mediante la property `text`.
    o explícitamente vía get_text()/set_text(). Las regiones definidas
    por el usuario también viven en ese XML; ver get_regions()/set_regions().

    Las visualizaciones cacheadas de la segmentación de líneas viven en
        {MEDIA_ROOT}/segmentation/{document_id}/page_{order:03d}_lines.jpg
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='pages')
    order    = models.PositiveIntegerField(verbose_name='Orden')
    image    = models.ImageField(upload_to=page_image_path, verbose_name='Facsimilar')

    # ── Estado del OCR (para el flujo progresivo) ───────────────────────
    OCR_PENDING    = 'pending'
    OCR_PROCESSING = 'processing'
    OCR_DONE       = 'done'
    OCR_ERROR      = 'error'
    OCR_STATUS_CHOICES = [
        (OCR_PENDING,    'Pendiente'),
        (OCR_PROCESSING, 'Procesando'),
        (OCR_DONE,       'Listo'),
        (OCR_ERROR,      'Error'),
    ]
    ocr_status = models.CharField(
        max_length=16, choices=OCR_STATUS_CHOICES, default=OCR_DONE,
        verbose_name='Estado OCR',
        help_text='Estado del procesado OCR para esta página. Las páginas '
                  'recién subidas empiezan en "pending"; el worker en '
                  'background las mueve a "processing" y luego a "done" '
                  '(o "error"). El frontend usa esto para bloquear la '
                  'navegación hasta que la página esté lista.',
    )
    ocr_error = models.TextField(blank=True, default='',
                                 verbose_name='Error OCR')

    def __str__(self):
        return f'{self.document.title} — p. {self.order}'

    # ── API basada en ficheros (transcripts XML) ────────────────────────

    def get_text(self) -> str:
        """Devuelve el texto transcrito leído del fichero XML. '' si no existe."""
        from .transcripts import get_text
        return get_text(self.document_id, self.order)

    def set_text(self, value: str) -> None:
        """
        Guarda el texto transcrito en el fichero XML, conservando regiones,
        metadatos y created_at si ya existían.
        """
        from .transcripts import save_text
        save_text(
            self.document_id,
            self.order,
            value or "",
            facsimile=self.image.name.split('/')[-1] if self.image else "",
            title=self.document.title or "",
            author=self.document.author or "",
            year=self.document.year,
            doc_type=self.document.document_type or "",
        )

    # `text` como property de compatibilidad con el código existente.
    text = property(fget=get_text, fset=set_text)

    def get_regions(self):
        """Devuelve la lista de regiones definidas por el usuario en esta página."""
        from .transcripts import get_regions
        return get_regions(self.document_id, self.order)

    def set_regions(self, regions) -> None:
        """Guarda las regiones (iterable de transcripts.Region) en el XML."""
        from .transcripts import save_regions
        save_regions(self.document_id, self.order, regions)

    class Meta:
        ordering        = ['order']
        unique_together = ['document', 'order']
        verbose_name    = 'Página'
        verbose_name_plural = 'Páginas'


# ── Operation log ──────────────────────────────────────────────────────────

class OperationLog(models.Model):
    INSERT_DOC       = 'insert_doc'
    EDIT_DOC         = 'edit_doc'
    VIEW_DOC         = 'view_doc'
    DOWNLOAD_DOC     = 'download_doc'
    INSERT_USER      = 'insert_user'
    REMOVE_USER      = 'remove_user'
    EDIT_PRIVILEGES  = 'edit_privileges'
    LOGIN            = 'login'
    LOGOUT           = 'logout'
    DELETE_DOC       = 'delete_doc'

    ACTION_CHOICES = [
        (INSERT_DOC,      'Inserción de documento'),
        (EDIT_DOC,        'Edición de documento'),
        (VIEW_DOC,        'Visualización de documento'),
        (DOWNLOAD_DOC,    'Descarga de documento'),
        (INSERT_USER,     'Inserción de usuario'),
        (REMOVE_USER,     'Eliminación de usuario'),
        (EDIT_PRIVILEGES, 'Edición de privilegios'),
        (LOGIN,           'Inicio de sesión'),
        (LOGOUT,          'Cierre de sesión'),
        (DELETE_DOC,      'Eliminación de documento'),
    ]

    user        = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, verbose_name='Usuario')
    action      = models.CharField(max_length=30, choices=ACTION_CHOICES, verbose_name='Acción')
    description = models.TextField(verbose_name='Descripción')
    timestamp   = models.DateTimeField(auto_now_add=True, verbose_name='Fecha y hora')

    def __str__(self):
        return f'{self.user} | {self.get_action_display()} | {self.timestamp:%d/%m/%Y %H:%M}'

    class Meta:
        ordering            = ['-timestamp']
        verbose_name        = 'Registro de operación'
        verbose_name_plural = 'Registros de operaciones'
