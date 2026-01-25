#!/bin/bash
source /home/orangepi/MonkeySee/venv/bin/activate
cd /home/orangepi/MonkeySee/
echo "Starting MonkeySee Dashboard (Production Mode)"
echo "Loading configuration..."
while :; do
python main.py
done
echo "MonkeySee exited"

