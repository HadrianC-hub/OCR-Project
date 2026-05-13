from django import forms
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
from .models import CustomUser

_INPUT = 'form-control'
_SELECT = 'form-select'
_CHECK = 'form-check-input'

# Roles disponibles al crear/editar usuarios — excluye Administrador Principal
_ROLE_CHOICES = [
    (CustomUser.WORKER, 'Trabajador'),
    (CustomUser.ADMIN,  'Administrador'),
]


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        label='Usuario',
        widget=forms.TextInput(attrs={'class': _INPUT, 'placeholder': 'Nombre de usuario', 'autofocus': True}),
    )
    password = forms.CharField(
        label='Contraseña',
        widget=forms.PasswordInput(attrs={'class': _INPUT, 'placeholder': 'Contraseña'}),
    )


class AddUserForm(forms.ModelForm):
    password = forms.CharField(
        label='Contraseña',
        min_length=8,
        widget=forms.PasswordInput(attrs={'class': _INPUT}),
    )
    password_confirm = forms.CharField(
        label='Confirmar contraseña',
        widget=forms.PasswordInput(attrs={'class': _INPUT}),
    )

    class Meta:
        model = CustomUser
        fields = ['username', 'first_name', 'last_name', 'email', 'role']
        widgets = {
            'username':   forms.TextInput(attrs={'class': _INPUT}),
            'first_name': forms.TextInput(attrs={'class': _INPUT}),
            'last_name':  forms.TextInput(attrs={'class': _INPUT}),
            'email':      forms.EmailInput(attrs={'class': _INPUT}),
            'role':       forms.Select(attrs={'class': _SELECT}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['role'].choices = _ROLE_CHOICES

    def clean(self):
        cleaned = super().clean()
        pwd  = cleaned.get('password')
        conf = cleaned.get('password_confirm')
        if pwd and conf and pwd != conf:
            raise forms.ValidationError('Las contraseñas no coinciden.')
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
        return user


class EditUserRoleForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['role']
        widgets = {'role': forms.Select(attrs={'class': _SELECT})}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['role'].choices = _ROLE_CHOICES


class CustomPasswordChangeForm(PasswordChangeForm):
    old_password = forms.CharField(
        label='Contraseña actual',
        widget=forms.PasswordInput(attrs={'class': _INPUT, 'autofocus': True}),
    )
    new_password1 = forms.CharField(
        label='Nueva contraseña',
        min_length=8,
        widget=forms.PasswordInput(attrs={'class': _INPUT}),
    )
    new_password2 = forms.CharField(
        label='Confirmar nueva contraseña',
        widget=forms.PasswordInput(attrs={'class': _INPUT}),
    )
