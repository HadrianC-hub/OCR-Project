"""
python manage.py reindex_search

Reconstruye desde cero el índice de búsqueda semántica.

Útil cuando:
  - Acabas de instalar el feature por primera vez
  - Has cambiado el modelo de embeddings
  - El store se corrompió y quieres rehacerlo
  - Tienes un volumen grande y prefieres indexar de una sentada
    en lugar de página a página (batching más eficiente)
"""

from django.core.management.base import BaseCommand, CommandError

from apps.search.indexer import reindex_all
from apps.search import encoder


class Command(BaseCommand):
    help = "Reconstruye el índice de búsqueda semántica desde cero."

    def add_arguments(self, parser):
        parser.add_argument(
            "--quiet", action="store_true",
            help="No imprimir progreso por stdout.",
        )

    def handle(self, *args, **opts):
        if not encoder.is_available():
            raise CommandError(
                "El modelo E5 no está disponible. Comprueba INSTALL.md: "
                "necesitas (a) instalar sentence-transformers y "
                "(b) tener el modelo en models/multilingual-e5-small/."
            )

        verbose = not opts["quiet"]
        if verbose:
            self.stdout.write("Reindexando el índice de búsqueda semántica...")
        stats = reindex_all(verbose=verbose)
        self.stdout.write(self.style.SUCCESS(
            f"Indexados {stats['n_chunks']} chunks "
            f"de {stats['n_pages']} páginas en {stats['n_docs']} documentos "
            f"({stats['elapsed_s']} s)."
        ))
