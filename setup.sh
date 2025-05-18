#!/bin/bash

echo "🚀 [1/6] Setting up Git identity..."
git config --global user.name "Jayaprakash"
git config --global user.email "jayprakeshsai@gmail.com"
git config --global credential.helper store

echo "📦 [2/6] Installing Python requirements..."
pip install -r requirements.txt

echo "🧠 [3/6] Installing development tools for IntelliSense..."
pip install ipykernel python-dotenv

echo "📁 [4/6] Creating VS Code config folder..."
mkdir -p .vscode

echo "⚙️ [5/6] Writing settings.json with IntelliSense configuration..."
cat <<EOF > .vscode/settings.json
{
  "python.languageServer": "Pylance",
  "python.analysis.indexing": true,
  "python.analysis.autoSearchPaths": true,
  "python.analysis.useLibraryCodeForTypes": true,
  "python.defaultInterpreterPath": "/usr/bin/python3"
}
EOF

echo "🧩 [6/6] Adding recommended extensions..."
cat <<EOF > .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "esbenp.prettier-vscode"
  ]
}
EOF

echo "✅ Setup complete!"
echo "💡 Final step: In VS Code, press Cmd+Shift+P → 'Reload Window' to apply IntelliSense and extension settings."
echo "💡 Also go to Cmd+Shift+P → 'Python: Select Interpreter' and choose: /usr/bin/python3 (Remote)"

# chmod +x setup.sh
# ./setup.sh
# huggingface-cli login