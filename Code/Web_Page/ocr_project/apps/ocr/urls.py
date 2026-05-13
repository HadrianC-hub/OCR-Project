from django.urls import path
from . import views

urlpatterns = [
    # OCR sobre el documento completo / página completa
    path('documents/<int:doc_id>/ocr/', views.ocr_process,    name='ocr_process'),
    path('documents/<int:doc_id>/ocr/status/',
         views.document_ocr_status, name='document_ocr_status'),
    path('ocr/page/<int:page_id>/rerun/', views.ocr_single_page, name='ocr_single_page'),

    # Visualización de la segmentación de líneas (JPG cacheado)
    path('ocr/page/<int:page_id>/segmentation.jpg',
         views.line_segmentation_image, name='line_segmentation_image'),

    # Cajas brutas de líneas/bloques en JSON (para el frontend)
    path('ocr/page/<int:page_id>/segmentation.json',
         views.line_segmentation_boxes, name='line_segmentation_boxes'),

    # OCR sobre las regiones definidas por el usuario en una página
    path('ocr/page/<int:page_id>/regions/rerun/',
         views.ocr_regions_page, name='ocr_regions_page'),
]
