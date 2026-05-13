from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    WORKER = 'worker'
    ADMIN = 'admin'
    SUPER_ADMIN = 'super_admin'

    ROLE_CHOICES = [
        (WORKER, 'Trabajador'),
        (ADMIN, 'Administrador'),
        (SUPER_ADMIN, 'Administrador Principal'),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default=WORKER,
        verbose_name='Rol',
    )

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def is_worker_or_above(self):
        """True for workers, admins and super-admins."""
        return self.role in [self.WORKER, self.ADMIN, self.SUPER_ADMIN]

    @property
    def is_admin_or_above(self):
        """True for admins and super-admins."""
        return self.role in [self.ADMIN, self.SUPER_ADMIN]

    @property
    def is_super_admin(self):
        return self.role == self.SUPER_ADMIN

    def get_role_display_name(self):
        return dict(self.ROLE_CHOICES).get(self.role, 'Desconocido')

    class Meta:
        verbose_name = 'Usuario'
        verbose_name_plural = 'Usuarios'
