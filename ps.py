from pathlib import Path

# Konfigurasi path dan struktur folder untuk project Xiaomi Sentiment Analysis
PROJECT_ROOT = Path(".").resolve()

# Folder utama output / artifacts
DUNGEON_DIR = PROJECT_ROOT / "dungeon"

# Struktur folder
DIRS = {
    "docs": PROJECT_ROOT / "docs",
    
    "dungeon_root": DUNGEON_DIR,
    "raw_dataset": DUNGEON_DIR / "raw_dataset",
    "cleaned_dataset": DUNGEON_DIR / "cleaned_dataset",
    "processed_dataset": DUNGEON_DIR / "processed_dataset",
    "baseline_hybrid": DUNGEON_DIR / "baseline_hybrid",
    "manual_labeling": DUNGEON_DIR / "manual_labeling",
    "setfit_finetune": DUNGEON_DIR / "setfit_finetune",
    "active_learning": DUNGEON_DIR / "active_learning",
    "final_model": DUNGEON_DIR / "final_model",
    "summary_analysis": DUNGEON_DIR / "summary_analysis",
}

# Subfolder tambahan agar lebih rapi
SUBDIRS = {
    "baseline_reports": DUNGEON_DIR / "baseline_hybrid" / "reports",
    "baseline_checkpoints": DUNGEON_DIR / "baseline_hybrid" / "checkpoints",
    "baseline_figures": DUNGEON_DIR / "baseline_hybrid" / "figures",

    "setfit_reports": DUNGEON_DIR / "setfit_finetune" / "reports",
    "setfit_checkpoints": DUNGEON_DIR / "setfit_finetune" / "checkpoints",
    "setfit_figures": DUNGEON_DIR / "setfit_finetune" / "figures",

    "al_rounds": DUNGEON_DIR / "active_learning" / "rounds",
    "al_reports": DUNGEON_DIR / "active_learning" / "reports",
    "al_figures": DUNGEON_DIR / "active_learning" / "figures",

    "final_model_files": DUNGEON_DIR / "final_model" / "model_files",
    "final_model_reports": DUNGEON_DIR / "final_model" / "reports",

    "summary_reports": DUNGEON_DIR / "summary_analysis" / "reports",
    "summary_figures": DUNGEON_DIR / "summary_analysis" / "figures",
    "summary_exports": DUNGEON_DIR / "summary_analysis" / "exports",
}

ALL_DIRS = {**DIRS, **SUBDIRS}

# Membuat semua folder yang diperlukan
for name, path in ALL_DIRS.items():
    path.mkdir(parents=True, exist_ok=True)

    # Tambahkan .gitkeep agar folder kosong tetap tersimpan di git
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

# README mini di dalam dungeon
dungeon_readme = DUNGEON_DIR / "README.md"
if not dungeon_readme.exists():
    dungeon_readme.write_text(
        """# Dungeon Directory

Folder ini digunakan untuk menyimpan seluruh output project, termasuk:
- raw dataset
- cleaned dataset
- processed dataset
- hasil baseline hybrid
- manual labeling
- hasil fine-tuning SetFit
- hasil active learning
- final model
- summary analysis

Tujuan utama:
1. Menjaga struktur project tetap rapi
2. Memudahkan tracking eksperimen
3. Memisahkan source code dari output/artifacts
""",
        encoding="utf-8"
    )

# PRINT SUMMARY
print("Struktur folder berhasil dibuat.\n")
print(f"Project root : {PROJECT_ROOT}")
print(f"Dungeon root : {DUNGEON_DIR}\n")

print("Daftar folder utama:")
for key in [
    "docs",
    "raw_dataset",
    "cleaned_dataset",
    "processed_dataset",
    "baseline_hybrid",
    "manual_labeling",
    "setfit_finetune",
    "active_learning",
    "final_model",
    "summary_analysis",
]:
    print(f"- {ALL_DIRS[key]}")

# TREE PREVIEW
print("\nPreview struktur:")
print("""
Xiaomi Sentiment Analysis
│
├── docs
│
├── dungeon
│   ├── raw_dataset
│   ├── cleaned_dataset
│   ├── processed_dataset
│   ├── baseline_hybrid
│   │   ├── reports
│   │   ├── checkpoints
│   │   └── figures
│   ├── manual_labeling
│   ├── setfit_finetune
│   │   ├── reports
│   │   ├── checkpoints
│   │   └── figures
│   ├── active_learning
│   │   ├── rounds
│   │   ├── reports
│   │   └── figures
│   ├── final_model
│   │   ├── model_files
│   │   └── reports
│   ├── summary_analysis
│   │   ├── reports
│   │   ├── figures
│   │   └── exports
│   └── README.md
│
└── README.md
""")