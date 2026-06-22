#!/usr/bin/env bash
cd /home/baver/lynx || exit 1
source .venv/bin/activate
exec python lynx_gui.py
