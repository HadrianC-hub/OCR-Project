from django.urls import path
from . import views, region_views

urlpatterns = [
    path('',                                 views.home,             name='home'),
    path('search/',                          views.search,           name='search'),
    path('documents/<int:doc_id>/',          views.view_document,    name='view_document'),
    path('documents/<int:doc_id>/download/', views.download_document, name='download_document'),
    path('documents/insert/',                views.insert_document,  name='insert_document'),
    path('documents/<int:doc_id>/edit/',     views.edit_document,    name='edit_document'),
    path('documents/<int:doc_id>/abandon/',  views.abandon_document, name='abandon_document'),
    path('documents/<int:doc_id>/delete/',   views.delete_document,  name='delete_document'),

    # Regiones definidas por el usuario en la pantalla de edición.
    # Endpoint AJAX que persiste la lista de regiones en el XML.
    path('pages/<int:page_id>/regions/',     region_views.save_regions, name='save_regions'),
]
