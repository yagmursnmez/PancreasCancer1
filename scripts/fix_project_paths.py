"""
PancreasAI - Otomatik Dosya Yolu Güncelleyici
Proje farklı bir bilgisayara veya klasöre taşındığında .env ve config.json
içerisindeki dosya yollarını mevcut konuma göre otomatik günceller.
"""
import os
import json
from pathlib import Path

def fix_paths():
    # Proje kök dizinini otomatik tespit et
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root).replace("\\", "/")
    
    print(f"\n[DOSYA YOLU KONTROLU] Mevcut Proje Konumu: {root_str}")
    
    # 1. .env dosyasını güncelle
    env_file = project_root / ".env"
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("BASE_PATH="):
                new_lines.append(f"BASE_PATH={root_str}")
            elif line.startswith("nnUNet_raw="):
                new_lines.append(f"nnUNet_raw={root_str}/data/nnunet_raw")
            elif line.startswith("nnUNet_preprocessed="):
                new_lines.append(f"nnUNet_preprocessed={root_str}/data/nnunet_preprocessed")
            elif line.startswith("nnUNet_results="):
                new_lines.append(f"nnUNet_results={root_str}/data/nnunet_results")
            else:
                new_lines.append(line)
        env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"  [OK] .env konfigürasyon dosyası yeni konuma ayarlandı.")
    else:
        print("  [UYARI] .env dosyası bulunamadı, oluşturuluyor...")
        env_content = f"""BASE_PATH={root_str}
nnUNet_raw={root_str}/data/nnunet_raw
nnUNet_preprocessed={root_str}/data/nnunet_preprocessed
nnUNet_results={root_str}/data/nnunet_results
DATASET_ID=007
DATASET_NAME=Dataset007_Pancreas
NNUNET_CONFIG=2d
NNUNET_TRAINER=nnUNetTrainer
NNUNET_PLANNER=nnUNetPlannerResEncM
FOLD=0
FLASK_PORT=5000
FLASK_DEBUG=False
PANCREAS_DEBUG=True
MAX_UPLOAD_MB=8192
MAX_UPLOAD_FILES=5000
MODEL_TIMEOUT_SECONDS=1800
MODEL_CHECKPOINT=checkpoint_best.pth
CUDA_VISIBLE_DEVICES=0
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_REQUIRED=True
CUDA_MODULE_LOADING=LAZY
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
SHIM_MCCOMPAT_ENABLE_GPU=1
GPU_EVIDENCE_REQUIRED=True
GPU_TELEMETRY_INTERVAL_SECONDS=1.0
"""
        env_file.write_text(env_content, encoding="utf-8")
        print(f"  [OK] .env dosyası oluşturuldu.")

    # 2. config.json dosyasını güncelle
    config_file = project_root / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "paths" in data and isinstance(data["paths"], dict):
                data["paths"]["base"] = root_str
                data["paths"]["data_raw"] = f"{root_str}/data/raw"
                data["paths"]["nnunet_raw"] = f"{root_str}/data/nnunet_raw"
                data["paths"]["nnunet_preprocessed"] = f"{root_str}/data/nnunet_preprocessed"
                data["paths"]["nnunet_results"] = f"{root_str}/data/nnunet_results"
                data["paths"]["inference_output"] = f"{root_str}/data/inference_output"
                data["paths"]["web_uploads"] = f"{root_str}/web/static/uploads"
                data["paths"]["web_results"] = f"{root_str}/web/static/results"
                data["paths"]["logs"] = f"{root_str}/logs"
                data["paths"]["metrics"] = f"{root_str}/metrics"
                
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"  [OK] config.json konfigürasyonu yeni konuma ayarlandı.")
        except Exception as e:
            print(f"  [UYARI] config.json güncellenirken hata: {e}")

if __name__ == "__main__":
    fix_paths()
