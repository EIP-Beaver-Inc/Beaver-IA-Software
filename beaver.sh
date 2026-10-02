#!/bin/bash
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "Erreur : Python introuvable."; exit 1; }
if ! "$PY" -c "import tkinter" 2>/dev/null; then
    echo "Erreur : tkinter manquant."
    echo "  Debian/Ubuntu : sudo apt install python3-tk"
    echo "  Fedora        : sudo dnf install python3-tkinter"
    exit 1
fi
exec "$PY" beaver.py "$@"
