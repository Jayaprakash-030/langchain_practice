#!/bin/bash

echo "🚀 [1/5] Setting up Git identity..."
git config --global user.name "Jayaprakash"
git config --global user.email "jayprakeshsai@gmail.com"
git config --global credential.helper store

echo "🔁 [2/5] Cloning repo if it doesn't exist..."
if [ ! -d langchain_practice ]; then
  git clone https://github.com/Jayaprakash-030/langchain_practice.git
else
  echo "✅ langchain_practice already exists. Pulling latest changes..."
  cd langchain_practice && git pull && cd ..
fi

echo "📦 [3/5] Installing requirements..."
cd langchain_practice || exit
pip install -r requirements.txt

echo "✅ [4/5] Verifying GPU..."
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# chmod +x setup.sh
# ./setup.sh

