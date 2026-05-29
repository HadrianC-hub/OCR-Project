from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

from .decorators import admin_required
from .forms import LoginForm, AddUserForm, EditUserRoleForm, CustomPasswordChangeForm
from .models import CustomUser


# ── Lazy import to avoid circular imports ──────────────────────────────────
def _log(user, action, description):
    from apps.documents.models import OperationLog
    OperationLog.objects.create(user=user, action=action, description=description)


# ── Auth ───────────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    form = LoginForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        _log(user, 'login', f'{user.username} inició sesión')
        return redirect(request.GET.get('next') or 'home')
    elif request.method == 'POST':
        messages.error(request, 'Usuario o contraseña incorrectos.')

    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    if request.user.is_authenticated:
        _log(request.user, 'logout', f'{request.user.username} cerró sesión')
        logout(request)
    return redirect('home')


# ── User management (admin+) ───────────────────────────────────────────────

@admin_required
def user_management(request):
    users = CustomUser.objects.filter(is_active=True).order_by('username')
    return render(request, 'accounts/user_management.html', {'users': users})


@admin_required
def add_user(request):
    form = AddUserForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        # Super-admin can create any role; regular admins can only create workers
        new_role = form.cleaned_data['role']
        if new_role in [CustomUser.ADMIN, CustomUser.SUPER_ADMIN] and not request.user.is_super_admin:
            messages.error(request, 'Solo el administrador principal puede crear administradores.')
            return render(request, 'accounts/add_user.html', {'form': form})

        user = form.save()
        _log(request.user, 'insert_user',
             f'{request.user.username} creó al usuario "{user.username}" con rol {user.get_role_display()}')
        messages.success(request, f'Usuario "{user.username}" creado correctamente.')
        return redirect('user_management')
    elif request.method == 'POST':
        messages.error(request, 'Corrige los errores del formulario.')

    return render(request, 'accounts/add_user.html', {'form': form})


@admin_required
def edit_user_role(request, user_id):
    target = get_object_or_404(CustomUser, pk=user_id, is_active=True)

    # Cannot edit super-admin
    if target.is_super_admin:
        messages.error(request, 'No se pueden modificar los privilegios del administrador principal.')
        return redirect('user_management')

    # An admin cannot modify their own role
    if target == request.user:
        messages.error(request, 'No puedes modificar tus propios privilegios.')
        return redirect('user_management')

    old_role = target.get_role_display()
    form = EditUserRoleForm(request.POST or None, instance=target)
    if request.method == 'POST' and form.is_valid():
        form.save()
        new_role = target.get_role_display()
        _log(request.user, 'edit_privileges',
             f'{request.user.username} cambió el rol de "{target.username}" de {old_role} a {new_role}')
        messages.success(request, f'Privilegios de "{target.username}" actualizados.')
        return redirect('user_management')

    return render(request, 'accounts/edit_user_role.html', {'form': form, 'target_user': target})


@admin_required
def delete_user(request, user_id):
    target = get_object_or_404(CustomUser, pk=user_id, is_active=True)

    if target.is_super_admin:
        messages.error(request, 'No se puede eliminar al administrador principal.')
        return redirect('user_management')
    if target == request.user:
        messages.error(request, 'No puedes eliminarte a ti mismo.')
        return redirect('user_management')

    if request.method == 'POST':
        username = target.username
        # Soft-delete: mantenemos la fila para preservar la integridad
        # del histórico (OperationLog y Document apuntan a este usuario
        # con on_delete=SET_NULL, así que un hard-delete vaciaría los
        # registros de autoría). Pero renombramos el username con un
        # sufijo único antes de desactivar, así el slot del nombre original
        # queda libre y un admin puede crear más tarde un usuario nuevo
        # con el mismo nombre (caso real cuando alguien se reincorpora
        # con un alias distinto, o cuando el nombre era genérico tipo
        # "biblioteca" o "becario"). El sufijo __deleted__<timestamp>
        # también sirve como marcador legible en el admin de Django.
        import time
        target.username = f"{username}__deleted__{int(time.time())}"
        target.is_active = False
        target.save()
        _log(request.user, 'remove_user',
             f'{request.user.username} eliminó al usuario "{username}"')
        messages.success(request, f'Usuario "{username}" eliminado del sistema.')
        return redirect('user_management')

    return render(request, 'accounts/confirm_delete_user.html', {'target_user': target})


# ── Change password ────────────────────────────────────────────────────────

def change_password(request):
    if not request.user.is_authenticated:
        return redirect('login')

    form = CustomPasswordChangeForm(request.user, request.POST or None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        # Keep the user logged in after password change
        from django.contrib.auth import update_session_auth_hash
        update_session_auth_hash(request, form.user)
        _log(request.user, 'change_password',
             f'{request.user.username} cambió su contraseña')
        messages.success(request, 'Contraseña actualizada correctamente.')
        return redirect('home')
    elif request.method == 'POST':
        messages.error(request, 'Corrige los errores del formulario.')

    return render(request, 'accounts/change_password.html', {'form': form})
