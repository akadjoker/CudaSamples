!/bin/bash

set -e 

echo "[1/7] Instalar dependências do sistema..."
#sudo apt update
#sudo apt install -y git cmake ninja-build gcc-9 g++-9 python3-dev libopenblas-dev libssl-dev

echo "[2/7] Criar ambiente virtual..."
python3 -m venv env-pytorch
source env-pytorch/bin/activate
pip install --upgrade pip setuptools

echo "[3/7] Clonar o PyTorch (v1.12.1)..."
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.12.1
git submodule sync
git submodule update --init --recursive

echo "[4/7] Definir ambiente de compilação..."
export CC=gcc-9
export CXX=g++-9
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="6.1"
export CUDA_HOME=/usr/local/cuda-11.4
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV:-$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')}"

echo "[5/7] Instalar dependências Python..."
pip install -r requirements.txt

echo "[6/7] Compilar o PyTorch com CUDA 11.4..."
python setup.py bdist_wheel

echo "[7/7] Instalar o pacote compilado..."
pip install dist/torch-*.whl

echo ""
echo "✅ PyTorch compilado e instalado com suporte a CUDA 11.4"
echo "👉 Ativa o ambiente com: source ~/env-pytorch-build/bin/activate"
echo "👉 Testa com:"
echo "    python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'"

