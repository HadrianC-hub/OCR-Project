"""
Instala todas las dependencias del proyecto desde requirements.txt.

Uso:
    python install_dependencies.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    req = Path(__file__).resolve().parent / 'requirements.txt'
    if not req.is_file():
        print(f'No se encontró requirements.txt en {req}')
        sys.exit(1)

    print('Instalando dependencias desde requirements.txt...')
    print()

    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-r', str(req)]
        )
    except subprocess.CalledProcessError:
        print()
        print('La instalación falló. Si estás en Python 3.13 + Windows y el')
        print('error proviene de sentencepiece intentando compilar desde')
        print('fuente, fuerza la versión con wheel precompilado:')
        print()
        print('    pip install "sentencepiece>=0.2.1"')
        print()
        print('Y vuelve a ejecutar este script.')
        sys.exit(1)

    print()
    print('Instalación completa.')


if __name__ == '__main__':
    main()
