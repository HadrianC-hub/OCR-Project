from django.contrib import admin
from .models import Document, Page, OperationLog


class PageInline(admin.TabularInline):
    model  = Page
    extra  = 0
    # `text` ya no es columna de DB (vive en transcripts/<doc_id>/page_NNN.xml).
    fields = ('order', 'image')


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display  = ('title', 'author', 'year', 'document_type', 'is_public',
                     'total_views', 'created_at')
    list_filter   = ('document_type', 'is_public')
    search_fields = ('title', 'author')
    inlines       = [PageInline]


@admin.register(OperationLog)
class OperationLogAdmin(admin.ModelAdmin):
    list_display    = ('user', 'action', 'description', 'timestamp')
    list_filter     = ('action',)
    search_fields   = ('user__username', 'description')
    readonly_fields = ('user', 'action', 'description', 'timestamp')
