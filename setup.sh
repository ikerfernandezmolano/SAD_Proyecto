#!/bin/bash

# Crear venv si no existe
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activar venv
source venv/bin/activate

# Instalar dependencias si existe requirements.txt
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

echo "Entorno listo y activado"