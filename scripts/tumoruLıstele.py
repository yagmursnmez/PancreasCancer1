from pathlib import Path
import SimpleITK as sitk
import numpy as np
 
labels_dir = Path(
    r"C:\Users\monster\Documents\PythonProjects\PancreasCancer\data\nnunet_raw\Dataset007_Pancreas\labelsTr"
)
 
files = sorted(labels_dir.glob("*.nii.gz"))
 
print("Toplam label dosyası:", len(files))
 
tumorlu = []
tumorsuz = []
beklenmeyen = []
 
for file_path in files:
    image = sitk.ReadImage(str(file_path))
    array = sitk.GetArrayFromImage(image)
 
    values = np.unique(array).tolist()
 
    if 2 in values:
        tumorlu.append(file_path.name)
    else:
        tumorsuz.append(file_path.name)
 
    if not set(values).issubset({0, 1, 2}):
        beklenmeyen.append((file_path.name, values))
 
print("Tümörlü:", len(tumorlu))
print("Tümörsüz:", len(tumorsuz))
print("Beklenmeyen etiketli:", len(beklenmeyen))
 
print("\nİlk 10 dosyanın değerleri:")
for file_path in files[:10]:
    image = sitk.ReadImage(str(file_path))
    array = sitk.GetArrayFromImage(image)
    print(file_path.name, np.unique(array).tolist())