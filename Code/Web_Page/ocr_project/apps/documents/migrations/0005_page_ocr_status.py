from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0004_document_use_bert_correction'),
    ]

    operations = [
        migrations.AddField(
            model_name='page',
            name='ocr_status',
            field=models.CharField(
                choices=[
                    ('pending',    'Pendiente'),
                    ('processing', 'Procesando'),
                    ('done',       'Listo'),
                    ('error',      'Error'),
                ],
                default='done',
                help_text=(
                    'Estado del procesado OCR para esta página. Las páginas '
                    'recién subidas empiezan en "pending"; el worker en '
                    'background las mueve a "processing" y luego a "done" '
                    '(o "error"). El frontend usa esto para bloquear la '
                    'navegación hasta que la página esté lista.'
                ),
                max_length=16,
                verbose_name='Estado OCR',
            ),
        ),
        migrations.AddField(
            model_name='page',
            name='ocr_error',
            field=models.TextField(blank=True, default='', verbose_name='Error OCR'),
        ),
    ]
