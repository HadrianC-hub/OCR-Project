from django import forms
from .models import Document

_INPUT  = 'form-control'
_SELECT = 'form-select'
_CHECK  = 'form-check-input'


class DocumentMetadataForm(forms.ModelForm):
    class Meta:
        model  = Document
        fields = ['title', 'year', 'author', 'description', 'document_type',
                  'is_public', 'use_bert_correction']
        widgets = {
            'title':         forms.TextInput(attrs={'class': _INPUT}),
            'year':          forms.NumberInput(attrs={'class': _INPUT, 'min': 1, 'max': 2100}),
            'author':        forms.TextInput(attrs={'class': _INPUT}),
            'description':   forms.Textarea(attrs={'class': _INPUT, 'rows': 3}),
            'document_type': forms.Select(attrs={'class': _SELECT}),
            'is_public':     forms.CheckboxInput(attrs={'class': _CHECK}),
            'use_bert_correction': forms.CheckboxInput(attrs={'class': _CHECK}),
        }


class SearchForm(forms.Form):
    q = forms.CharField(
        required=False, label='Búsqueda',
        widget=forms.TextInput(attrs={
            'class': _INPUT + ' form-control-lg',
            'placeholder': 'Buscar por título, autor o descripción…',
            'autocomplete': 'off',
        }),
    )
    year_from = forms.IntegerField(
        required=False, label='Año desde',
        widget=forms.NumberInput(attrs={'class': _INPUT, 'placeholder': 'Desde'}),
    )
    year_to = forms.IntegerField(
        required=False, label='Año hasta',
        widget=forms.NumberInput(attrs={'class': _INPUT, 'placeholder': 'Hasta'}),
    )
    document_type = forms.ChoiceField(
        required=False, label='Tipo',
        choices=[('', 'Todos'), ('printed', 'Impreso'), ('manuscript', 'Manuscrito')],
        widget=forms.Select(attrs={'class': _SELECT}),
    )
    author = forms.CharField(
        required=False, label='Autor',
        widget=forms.TextInput(attrs={'class': _INPUT, 'placeholder': 'Filtrar por autor'}),
    )
