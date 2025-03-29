#!/bin/bash

set -e

echo "🔧 Criar ambiente virtual em ~/env-cu113..."
python3 -m venv env-cu
source env-cu/bin/activate

echo "⬆️ Atualizar pip e instalar PyTorch com suporte a CUDA 11.3..."
pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1   --extra-index-url https://download.pytorch.org/whl/cu113


echo ""
echo "✅ Ambiente configurado! Para voltar a usá-lo depois:"
echo "    source env-cu/bin/activate"

