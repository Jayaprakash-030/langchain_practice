#!/bin/bash

echo "🚀 [1/4] Setting up Git identity..."
git config --global user.name "Jayaprakash"
git config --global user.email "jayprakeshsai@gmail.com"
git config --global credential.helper store

echo "📦 [2/4] Installing requirements..."
pip install -r requirements.txt

echo "✅ [3/4] Verifying GPU..."
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# chmod +x setup.sh
# ./setup.sh
