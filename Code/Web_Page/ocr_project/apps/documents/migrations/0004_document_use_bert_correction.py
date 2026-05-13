from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0003_text_to_xml_files'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='use_bert_correction',
            field=models.BooleanField(
                default=False,
                help_text='Usa el reranker neuronal BETO para escoger entre los '
                          'candidatos de SymSpell. Mucho más preciso en casos '
                          'ambiguos pero 100x más lento (1-5 s por página). '
                          'Recomendado para documentos donde la calidad importe '
                          'más que la velocidad.',
                verbose_name='Corrección con BERT',
            ),
        ),
    ]
