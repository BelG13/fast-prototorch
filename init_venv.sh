#!/bin/bash

# Define the virtual env folder path
FILE="./venv"

# Check if the file exists and is a regular file
if [ -d "$FILE" ]; then
    pip install -r requirements.txt --quiet
    echo "Packages installed"
else
    echo "virtualenv folder not detected, creation of one called venv"
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
    echo "Packages installed"
fi
