#!/bin/bash
source /home/orangepi/MonkeySee/venv/bin/activate
cd /home/orangepi/MonkeySee/
echo "Starting MonkeySee Dashboard (Production Mode)"
echo "Loading configuration..."
python main.py
echo "MonkeySee exited"

