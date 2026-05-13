from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages


def worker_required(view_func):
    """Requires the user to be authenticated and have at least 'worker' role."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, 'Debes iniciar sesión para acceder a esta página.')
            return redirect('login')
        if not request.user.is_worker_or_above:
            messages.error(request, 'No tienes permisos suficientes.')
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return wrapper


def admin_required(view_func):
    """Requires the user to be authenticated and have at least 'admin' role."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, 'Debes iniciar sesión para acceder a esta página.')
            return redirect('login')
        if not request.user.is_admin_or_above:
            messages.error(request, 'No tienes permisos de administrador.')
            return redirect('home')
        return view_func(request, *args, **kwargs)
    return wrapper
