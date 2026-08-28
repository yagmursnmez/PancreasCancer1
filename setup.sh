#!/bin/bash
# ============================================================
# Pancreas Cancer Detection - Linux/Mac Kurulum Scripti
# ============================================================

set -e  # Hata durumunda dur

echo "[1/6] Python sanal ortamı oluşturuluyor..."
python3 -m venv venv

echo "[2/6] Sanal ortam aktif ediliyor..."
source venv/bin/activate

echo "[3/6] pip güncelleniyor..."
pip install --upgrade pip

echo "[4/6] PyTorch kuruluyor (CUDA 11.8)..."
# !! BU SATIRI DEĞİŞTİR: CUDA sürümünüze göre değiştirin !!
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1 (yorumu kaldır):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU (GPU yoksa):
# pip install torch torchvision

echo "[5/6] Diğer gereksinimler kuruluyor..."
pip install -r requirements.txt

echo "[6/6] nnU-Net ortam değişkenleri ayarlanıyor..."
# !! BU SATIRI DEĞİŞTİR: Kendi proje yolunuzu girin !!
export nnUNet_raw="$(pwd)/data/nnunet_raw"
export nnUNet_preprocessed="$(pwd)/data/nnunet_preprocessed"
export nnUNet_results="$(pwd)/data/nnunet_results"

# .bashrc veya .zshrc'ye ekle (kalıcı)
echo "export nnUNet_raw='$(pwd)/data/nnunet_raw'" >> ~/.bashrc
echo "export nnUNet_preprocessed='$(pwd)/data/nnunet_preprocessed'" >> ~/.bashrc
echo "export nnUNet_results='$(pwd)/data/nnunet_results'" >> ~/.bashrc

echo ""
echo "============================================================"
echo "KURULUM TAMAMLANDI!"
echo "============================================================"
echo "Ortam aktif etmek için: source venv/bin/activate"
echo "Kurulumu doğrulamak için: python setup_project.py"
