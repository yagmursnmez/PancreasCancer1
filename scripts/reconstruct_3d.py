"""
============================================================
ADIM 8: 3D REKONSTRÜKSIYON
============================================================
Bu script:
1. Segmentasyon maskesini 3D yüzeye dönüştürür
   (Marching Cubes algoritması)
2. Pankreas ve tümör için ayrı 3D mesh oluşturur
3. STL ve OBJ formatında kaydeder
4. 3D görselleştirme PNG oluşturur
5. Web'de gösterilebilir interactive HTML dosyası oluşturur

Kullanım:
    python scripts/reconstruct_3d.py
    python scripts/reconstruct_3d.py --mask data/inference_output/segmentation_masks/case_0001.nii.gz
============================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np

# ============================================================
# PATH AYARLARI
# ⚠️ BU SATIRI DEĞİŞTİR
# ============================================================
BASE_PATH = Path(__file__).parent.parent

RECON_3D_DIR = BASE_PATH / "data" / "inference_output" / "3d_reconstructions"
VIZ_DIR      = BASE_PATH / "data" / "inference_output" / "visualizations"

# ============================================================
# RENK KODLARI
# ============================================================
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[OK]{RESET}   {msg}")
def warn(msg): print(f"  {YELLOW}[!!]{RESET}   {msg}")
def fail(msg): print(f"  {RED}[ERR]{RESET}  {msg}")
def info(msg): print(f"  {CYAN}[--]{RESET}   {msg}")
def header(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


# ============================================================
# BÖLÜM 1: MASKE YÜKLEME VE HAZIRLAMA
# ============================================================
def load_and_prepare_mask(
    mask_path: Path,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[np.ndarray, Tuple]:
    """
    Segmentasyon maskesini yükler ve 3D yüzey için hazırlar.

    Args:
        mask_path: .nii.gz maske dosyası
        spacing:   Voxel spacing (mm). None ise NIfTI'den okunur.

    Returns:
        (mask_data, voxel_spacing)
    """
    header("MASKE YUKLENIYOR")

    try:
        import nibabel as nib
        img  = nib.load(str(mask_path))
        data = img.get_fdata().astype(np.int32)

        # Voxel spacing
        if spacing is None:
            zooms = img.header.get_zooms()[:3]
            spacing = tuple(float(z) for z in zooms)
            if any(s == 0 for s in spacing):
                spacing = (1.0, 1.0, 1.0)  # Varsayılan

        ok(f"Maske yuklendi: {data.shape}")
        ok(f"Voxel spacing: {spacing} mm")

        # Etiket istatistikleri
        labels, counts = np.unique(data, return_counts=True)
        for lbl, cnt in zip(labels, counts):
            name = {0: "background", 1: "pancreas", 2: "tumor"}.get(lbl, f"label_{lbl}")
            info(f"  Etiket {lbl} ({name}): {cnt:,} voksel")

        return data, spacing

    except ImportError:
        fail("nibabel kurulu degil! pip install nibabel")
        return np.zeros((64, 64, 64)), (1., 1., 1.)
    except Exception as e:
        fail(f"Maske yuklenemedi: {e}")
        return np.zeros((64, 64, 64)), (1., 1., 1.)


# ============================================================
# BÖLÜM 2: MARCHING CUBES — 3D YÜzey ÇIKARMA
# ============================================================
def extract_surface_mesh(
    mask:    np.ndarray,
    label:   int,
    spacing: Tuple,
    smooth_iterations: int = 3,
    level:   float = 0.5,
    affine: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Marching Cubes algoritmasıyla 3D yüzey mesh'i çıkarır.

    Marching Cubes:
    - Voksel grid'ini üçgen mesh'e dönüştürür
    - level=0.5 → ikili maskenin yüzeyi
    - smooth_iterations: Laplacian smoothing

    Args:
        mask:              3D segmentasyon maskesi
        label:             Yüzeyi çıkarılacak etiket (1=pancreas, 2=tumor)
        spacing:           Voxel boyutu (mm)
        smooth_iterations: Smoothing tekrar sayısı
        level:             Marching Cubes eşiği

    Returns:
        (vertices, faces, normals) veya None
    """
    try:
        from skimage.measure import marching_cubes
        from skimage.filters import gaussian

        label_points = np.where(mask == label)
        if not label_points[0].size:
            info(f"  Etiket {label} icin voksel yok, yuzey cikarilmiyor.")
            return None

        # Tam CT hacminde Gaussian uygulamak yüzlerce MB geçici bellek ve gereksiz
        # süre tüketir. Yalnız etiketi çevreleyen güvenli kutuda çalış; koordinatı
        # daha sonra tekrar hacmin fiziksel başlangıcına taşı.
        padding = 3
        starts = np.asarray([
            max(0, int(axis.min()) - padding) for axis in label_points
        ], dtype=int)
        stops = np.asarray([
            min(mask.shape[index], int(axis.max()) + padding + 1)
            for index, axis in enumerate(label_points)
        ], dtype=int)
        bounds = tuple(slice(int(starts[i]), int(stops[i])) for i in range(3))
        binary_mask = (mask[bounds] == label).astype(np.float32)

        # Gaussian smoothing (daha düzgün yüzey için)
        if smooth_iterations > 0:
            sigma = 1.0 + smooth_iterations * 0.3
            binary_mask = gaussian(binary_mask, sigma=sigma)

        # Marching Cubes
        verts, faces, normals, _ = marching_cubes(
            binary_mask,
            level=level,
            spacing=spacing,
            allow_degenerate=True
        )
        spacing_array = np.asarray(spacing, dtype=np.float64)
        verts += starts.astype(np.float64) * spacing_array
        if affine is not None:
            affine_array = np.asarray(affine, dtype=np.float64)
            if affine_array.shape != (4, 4):
                raise ValueError("affine 4x4 olmalıdır")
            voxel_vertices = verts / spacing_array
            verts = (
                voxel_vertices @ affine_array[:3, :3].T
                + affine_array[:3, 3]
            )

        ok(f"Etiket {label}: {len(verts):,} vertex, {len(faces):,} uc gen")
        return verts, faces, normals

    except ImportError as e:
        warn(f"scikit-image bulunamadi: {e}")
        warn("pip install scikit-image")
        return None
    except Exception as e:
        warn(f"Marching Cubes hatasi (label {label}): {e}")
        return None


# ============================================================
# BÖLÜM 3: STL DOSYASI KAYDETME
# ============================================================
def save_as_stl(
    vertices: np.ndarray,
    faces:    np.ndarray,
    output_path: Path
) -> bool:
    """
    Mesh'i binary STL formatında kaydeder.

    STL formatı: 3D yazıcı ve görselleştirme yazılımları için.
    Paraview, MeshLab, Slicer 3D ile açılabilir.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_triangles = len(faces)

        with open(str(output_path), "wb") as f:
            import struct

            # STL header (80 byte)
            f.write(b"nnUNet Pankreas Segmentasyon" + b" " * 52)

            # Triangle sayısı (4 byte uint32)
            f.write(struct.pack("<I", n_triangles))

            # Her üçgen: normal (3 float) + 3 vertex (9 float) + attribute (2 byte)
            for tri in faces:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal /= norm_len

                f.write(struct.pack("<fff", *normal))
                f.write(struct.pack("<fff", *v0))
                f.write(struct.pack("<fff", *v1))
                f.write(struct.pack("<fff", *v2))
                f.write(struct.pack("<H", 0))  # attribute

        size_kb = output_path.stat().st_size / 1024
        ok(f"STL kaydedildi: {output_path.name} ({size_kb:.1f} KB, {n_triangles:,} uc gen)")
        return True

    except Exception as e:
        warn(f"STL kaydetme hatasi: {e}")
        return False


# ============================================================
# BÖLÜM 4: 3D VİZUALİZASYON (PNG)
# ============================================================
def render_3d_visualization(
    pancreas_mesh: Optional[Tuple],
    tumor_mesh:    Optional[Tuple],
    has_tumor:     bool,
    output_path:   Path
) -> bool:
    """
    Matplotlib ile 3D wireframe görselleştirmesi oluşturur.
    Pankreas (mavi) + Tümör (kırmızı) bileşik görüntü.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(16, 8), facecolor="#0d0d1a")

        views = [
            (30,  45,  "On Gorunum"),
            (30,  135, "Yan Gorunum"),
            (90,  0,   "Ust Gorunum"),
            (30,  -45, "3/4 Gorunum"),
        ]

        for i, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(2, 2, i+1, projection="3d")
            ax.set_facecolor("#0d0d1a")

            # Pankreas mesh
            if pancreas_mesh:
                verts, faces, _ = pancreas_mesh
                mesh = Poly3DCollection(
                    verts[faces],
                    alpha=0.3,
                    facecolor="#3498db",
                    edgecolor="none"
                )
                ax.add_collection3d(mesh)
                ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
                ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
                ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

            # Tümör mesh
            if tumor_mesh:
                verts_t, faces_t, _ = tumor_mesh
                tumor_poly = Poly3DCollection(
                    verts_t[faces_t],
                    alpha=0.8,
                    facecolor="#e74c3c",
                    edgecolor="#c0392b",
                    linewidth=0.2
                )
                ax.add_collection3d(tumor_poly)

            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, color="white", fontsize=9)
            ax.tick_params(colors="gray", labelsize=6)
            ax.grid(True, alpha=0.2)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor("gray")

        # Başlık
        title_color = "#e74c3c" if has_tumor else "#2ecc71"
        prediction  = "TUMOR VAR" if has_tumor else "TUMOR YOK"
        fig.suptitle(
            f"3D Pankreas Rekonstruksiyonu | {prediction}",
            fontsize=14, fontweight="bold",
            color=title_color
        )

        # Legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color="#3498db", alpha=0.5, label="Pankreas"),
            mpatches.Patch(color="#e74c3c", alpha=0.8, label="Tumor"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2,
                  facecolor="#1a1a2e", labelcolor="white", fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                   facecolor="#0d0d1a")
        plt.close()
        ok(f"3D gorsel kaydedildi: {output_path.name}")
        return True

    except ImportError as e:
        warn(f"Matplotlib/mpl_toolkits bulunamadi: {e}")
        return False
    except Exception as e:
        warn(f"3D render hatasi: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# BÖLÜM 5: INTERACTIVE HTML (Plotly)
# ============================================================
def create_interactive_html(
    pancreas_mesh: Optional[Tuple],
    tumor_mesh:    Optional[Tuple],
    has_tumor:     bool,
    output_path:   Path,
    coordinate_system: str = "voxel",
    tumor_core_mesh: Optional[Tuple] = None,
    tumor_envelope_mesh: Optional[Tuple] = None,
) -> bool:
    """
    Plotly ile interaktif 3D HTML görselleştirme oluşturur.
    Tarayıcıda döndürülebilir, zoom yapılabilir.
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Pankreas
        if pancreas_mesh:
            verts, faces, _ = pancreas_mesh
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name="Pankreas",
                color="#3498db",
                opacity=0.3,
                lighting=dict(ambient=0.5, diffuse=0.8),
            ))

        # Tümör
        if tumor_envelope_mesh:
            verts_e, faces_e, _ = tumor_envelope_mesh
            fig.add_trace(go.Mesh3d(
                x=verts_e[:, 0], y=verts_e[:, 1], z=verts_e[:, 2],
                i=faces_e[:, 0], j=faces_e[:, 1], k=faces_e[:, 2],
                name="Duyarlı belirsizlik sınırı",
                color="#f39c12",
                opacity=0.14,
                lighting=dict(ambient=0.7, diffuse=0.5),
            ))

        if tumor_mesh:
            verts_t, faces_t, _ = tumor_mesh
            fig.add_trace(go.Mesh3d(
                x=verts_t[:, 0], y=verts_t[:, 1], z=verts_t[:, 2],
                i=faces_t[:, 0], j=faces_t[:, 1], k=faces_t[:, 2],
                name="Kalibre ana tümör sınırı",
                color="#e74c3c",
                opacity=0.58 if tumor_core_mesh else 0.8,
                lighting=dict(ambient=0.5, diffuse=0.8),
            ))

        if tumor_core_mesh:
            verts_c, faces_c, _ = tumor_core_mesh
            fig.add_trace(go.Mesh3d(
                x=verts_c[:, 0], y=verts_c[:, 1], z=verts_c[:, 2],
                i=faces_c[:, 0], j=faces_c[:, 1], k=faces_c[:, 2],
                name="Yüksek güvenli çekirdek",
                color="#a93226",
                opacity=0.92,
                lighting=dict(ambient=0.45, diffuse=0.85),
            ))

        prediction = "TÜMÖR SEGMENTASYON ADAYI" if has_tumor else "TÜMÖR MASKESİ YOK"
        title_color = "red" if has_tumor else "green"
        if coordinate_system.upper() == "RAS":
            axis_titles = (
                "Sağ/Sol — RAS X (mm)",
                "Ön/Arka — RAS Y (mm)",
                "Alt/Üst — RAS Z (mm)",
            )
        else:
            axis_titles = ("Dizi ekseni 0 (mm)", "Dizi ekseni 1 (mm)", "Dizi ekseni 2 (mm)")

        fig.update_layout(
            title=dict(
                text=f"Model maskesinin 3B yüzeyi | {prediction} | Tanısal değildir",
                font=dict(size=16, color=title_color),
                x=0.5
            ),
            scene=dict(
                xaxis_title=axis_titles[0],
                yaxis_title=axis_titles[1],
                zaxis_title=axis_titles[2],
                aspectmode="data",
                bgcolor="rgb(13, 13, 26)",
                xaxis=dict(gridcolor="gray", showbackground=True,
                          backgroundcolor="rgb(20,20,40)"),
                yaxis=dict(gridcolor="gray", showbackground=True,
                          backgroundcolor="rgb(20,20,40)"),
                zaxis=dict(gridcolor="gray", showbackground=True,
                          backgroundcolor="rgb(20,20,40)"),
            ),
            paper_bgcolor="rgb(13, 13, 26)",
            font=dict(color="white"),
            legend=dict(bgcolor="rgb(30,30,60)", font=dict(color="white")),
            margin=dict(l=0, r=0, t=50, b=0),
            height=700,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        ok(f"Interactive HTML kaydedildi: {output_path.name}")
        return True

    except ImportError:
        warn("plotly bulunamadi: pip install plotly")
        return False
    except Exception as e:
        warn(f"HTML olusturma hatasi: {e}")
        return False


# ============================================================
# BÖLÜM 6: ANA İŞLEM
# ============================================================
def reconstruct_3d(mask_path: Path) -> Dict:
    """
    Tek bir maske dosyasını 3D'ye dönüştürür.
    """
    header(f"3D REKONSTRUKSIYON: {mask_path.name}")

    # Maske yükle
    mask_data, spacing = load_and_prepare_mask(mask_path)
    import nibabel as nib
    mask_affine = nib.load(str(mask_path)).affine

    # Pankreas ve tümör tespiti
    has_tumor = bool((mask_data == 2).sum() > 50)
    case_id   = mask_path.stem

    # Çıktı klasörü
    case_dir = RECON_3D_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "case_id":  case_id,
        "has_tumor": has_tumor,
        "prediction": "Tumor Var" if has_tumor else "Tumor Yok",
        "outputs":  {}
    }

    # 1. Marching Cubes — 3D yüzey çıkarma
    header("MARCHING CUBES - 3D YUZEY CIKARMA")
    pancreas_mesh = extract_surface_mesh(
        mask_data, label=1, spacing=spacing, smooth_iterations=3, affine=mask_affine
    )
    tumor_mesh = extract_surface_mesh(
        mask_data, label=2, spacing=spacing, smooth_iterations=2, affine=mask_affine
    )

    # 2. STL kaydet
    header("STL DOSYALARI KAYDEDILIYOR")
    if pancreas_mesh:
        stl_path = case_dir / f"{case_id}_pancreas.stl"
        if save_as_stl(pancreas_mesh[0], pancreas_mesh[1], stl_path):
            results["outputs"]["pancreas_stl"] = str(stl_path)

    if tumor_mesh and has_tumor:
        stl_path = case_dir / f"{case_id}_tumor.stl"
        if save_as_stl(tumor_mesh[0], tumor_mesh[1], stl_path):
            results["outputs"]["tumor_stl"] = str(stl_path)

    # 3. 3D görsel (PNG)
    header("3D GORSEL OLUSTURULUYOR")
    png_path = VIZ_DIR / f"{case_id}_3d_reconstruction.png"
    if render_3d_visualization(pancreas_mesh, tumor_mesh, has_tumor, png_path):
        results["outputs"]["3d_png"] = str(png_path)

    # 4. Interactive HTML
    html_path = case_dir / f"{case_id}_3d_interactive.html"
    if create_interactive_html(
        pancreas_mesh, tumor_mesh, has_tumor, html_path, coordinate_system="RAS"
    ):
        results["outputs"]["interactive_html"] = str(html_path)

    # 5. Web için kopyala
    web_results_dir = BASE_PATH / "web" / "static" / "results"
    web_results_dir.mkdir(parents=True, exist_ok=True)
    if png_path.exists():
        import shutil
        shutil.copy2(str(png_path), str(web_results_dir / png_path.name))
        results["outputs"]["web_png"] = str(web_results_dir / png_path.name)

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Segmentasyon maskesinden 3D rekonstrüksiyon")
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        # ⚠️ BU SATIRI DEĞİŞTİR: Tek maske dosyası yolu
        help="Segmentasyon maske dosyası (.nii.gz)"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=str(BASE_PATH / "data" / "inference_output" / "segmentation_masks"),
        help="Tüm maskelerin bulunduğu klasör"
    )
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}
==============================================================
  ADIM 8: 3D REKONSTRÜKSIYON
  Segmentasyon Maskesi -> 3D Yüzey (STL, PNG, HTML)
==============================================================
{RESET}
""")

    all_results = []

    # Tek dosya modu
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            fail(f"Maske bulunamadi: {mask_path}")
            return 1
        result = reconstruct_3d(mask_path)
        all_results.append(result)

    else:
        # Toplu mod
        mask_dir = Path(args.mask_dir)
        mask_files = sorted(mask_dir.glob("*.nii.gz")) if mask_dir.exists() else []

        if not mask_files:
            warn(f"Maske dosyası bulunamadı: {mask_dir}")
            warn("Simülasyon maskesi oluşturuluyor...")

            # Simülasyon maskesi oluştur ve işle
            sim_mask_path = BASE_PATH / "data" / "nnunet_raw" / "Dataset007_Pancreas" / "labelsTr"
            mask_files = sorted(sim_mask_path.glob("*.nii.gz"))[:3]  # İlk 3 dosya

            if not mask_files:
                warn("Hiç maske bulunamadı. Önce ADIM 2 ve 6'yı çalıştırın.")
                return 1

        info(f"Toplam {len(mask_files)} maske isleniyor...")
        for mask_path in mask_files:
            result = reconstruct_3d(mask_path)
            all_results.append(result)

    # Özet
    header("ADIM 8 TAMAMLANDI")
    tumor_count   = sum(1 for r in all_results if r.get("has_tumor"))
    healthy_count = len(all_results) - tumor_count

    ok(f"Islenen:    {len(all_results)} vaka")
    ok(f"Tumor Var:  {tumor_count}")
    ok(f"Tumor Yok:  {healthy_count}")
    ok(f"STL cikti:  {RECON_3D_DIR}")
    ok(f"PNG cikti:  {VIZ_DIR}")

    # JSON kaydet
    metrics_dir = BASE_PATH / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    report_path = metrics_dir / f"reconstruction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "step":    "ADIM8_3DReconstruction",
            "results": all_results
        }, f, indent=4, ensure_ascii=False)
    ok(f"Rapor: {report_path}")

    print(f"""
  {BOLD}Sonraki Adim (ADIM 9) - Web Arayuzu:{RESET}
    {CYAN}python web/app.py{RESET}
    Tarayici: http://localhost:5000
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
