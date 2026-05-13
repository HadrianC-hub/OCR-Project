"""
Filtros de template para apps.documents.

Uso en template:
    {% load search_filters %}
    {{ my_dict|get_item:my_key }}
"""

from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """
    Permite acceder a `dictionary[key]` desde un template, donde `key`
    es una variable. Necesario porque la sintaxis `{{ d.key }}` no
    funciona si `key` es dinámica.
    """
    if not isinstance(dictionary, dict):
        return None
    # Probamos con la clave tal cual y también convirtiendo tipos
    if key in dictionary:
        return dictionary[key]
    try:
        return dictionary[int(key)]
    except (ValueError, TypeError, KeyError):
        pass
    try:
        return dictionary[str(key)]
    except (KeyError, TypeError):
        pass
    return None
