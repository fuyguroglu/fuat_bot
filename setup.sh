#!/bin/bash
# Fuat_bot Setup Script for WSL
# Run this from the Fuat_bot directory

set -e

echo "🤖 Setting up Fuat_bot development environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'fuat_bot'..."
conda create -n fuat_bot python=3.11 -y

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  conda activate fuat_bot"
echo "  2. Install dependencies:      pip install -r requirements.txt"
echo "  3. Set up your API key:       cp .env.example .env && nano .env"
echo "  4. Run the agent:             python -m fuat_bot.cli"
echo ""
echo "🦞 Happy building!"
