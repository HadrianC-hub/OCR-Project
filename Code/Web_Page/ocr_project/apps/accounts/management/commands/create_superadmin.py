"""
Management command: create_superadmin

Creates the initial super-admin account that cannot be deleted through
the web interface.

Usage:
    python manage.py create_superadmin
    python manage.py create_superadmin --username admin --password secret
"""
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Crea el usuario administrador principal del sistema.'

    def add_arguments(self, parser):
        parser.add_argument('--username', default='superadmin',
                            help='Nombre de usuario (default: superadmin)')
        parser.add_argument('--password', default=None,
                            help='Contraseña (si no se indica se pedirá de forma interactiva)')
        parser.add_argument('--email', default='',
                            help='Email del administrador principal')

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        email    = options['email']

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(
                f'El usuario "{username}" ya existe. No se ha creado ningún usuario nuevo.'
            ))
            return

        if not password:
            import getpass
            password = getpass.getpass(f'Contraseña para "{username}": ')
            confirm  = getpass.getpass('Confirma la contraseña: ')
            if password != confirm:
                raise CommandError('Las contraseñas no coinciden.')

        if len(password) < 8:
            raise CommandError('La contraseña debe tener al menos 8 caracteres.')

        user = User.objects.create_superuser(
            username=username,
            password=password,
            email=email,
        )
        user.role = User.SUPER_ADMIN
        user.save()

        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Administrador principal "{username}" creado correctamente.\n'
            f'  Accede en: http://localhost:8000/login/\n'
        ))
