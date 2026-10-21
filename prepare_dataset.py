"""
Merge PlantVillage tomato (diseases) + AgriNet tomato (nutrient deficiency, pests)
into ONE folder tree that Keras can read:

tomato_combined/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg      (carved out of the validation images)

Originals are COPIED, never moved or deleted.
"""
import random
import shutil
from pathlib import Path

# ----------------------- EDIT THESE PATHS -----------------------
PV_ROOT = Path("dataset/PV_Tomato/tomato")  # PlantVillage folder that contains train/ and val/
AG_ROOT = Path("dataset/AG_Tomato/tomato")  # contains TRAINING/ and VALIDATION/
OUT = Path("tomato_combined")
# ----------------------------------------------------------------

TEST_FRACTION_OF_VAL = 0.5  # half of the validation images become the test set
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# AgriNet folder name (lowercase) -> unified class name.
# VERIFY these abbreviations against the dataset's documentation / class_distribution.xlsx.
AG_MAP = {
    "tomato__healthy": "Healthy",
    "tomato__n": "Nitrogen_deficiency",
    "tomato__k": "Potassium_deficiency",
    "tomato__n_k": "Nitrogen_Potassium_deficiency",
    "tomato__mit": "Mites",
    "tomato__lm": "Leaf_miner",
    "tomato__jas_mit": "Jassids_and_mites",
}


def pv_class_name(folder: str) -> str:
    """Tomato___Early_blight -> Early_blight, Tomato___healthy -> Healthy"""
    name = folder.replace("Tomato___", "").replace("Tomato__", "")
    if name.lower() == "healthy":
        return "Healthy"
    if name.lower().startswith("spider_mites"):
        return "Spider_mites"
    return name


def ag_class_name(folder: str) -> str:
    key = folder.lower()
    if key in AG_MAP:
        return AG_MAP[key]
    print(f"  [!] Unmapped AgriNet folder '{folder}' - keeping its own name. Add it to AG_MAP.")
    return folder.replace("tomato__", "").replace("tomato_", "")


def find_dir(parent: Path, wanted: str) -> Path:
    """Case-insensitive sub-folder lookup (TOMATO / Tomato / tomato)."""
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == wanted.lower():
            return p
    raise FileNotFoundError(f"No '{wanted}' folder inside {parent}")


def copy_split(src_split: Path, namer, prefix: str, out_split: str):
    for class_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
        dest = OUT / out_split / namer(class_dir.name)
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in class_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                # prefix avoids filename clashes between the two datasets
                shutil.copy2(f, dest / f"{prefix}_{f.name}")
                n += 1
        print(f"  {out_split:5s} {class_dir.name:45s} -> {dest.name:32s} {n}")


def make_test_split():
    rng = random.Random(SEED)
    for class_dir in sorted((OUT / "val").iterdir()):
        files = sorted(class_dir.glob("*"))
        rng.shuffle(files)
        n_test = int(len(files) * TEST_FRACTION_OF_VAL)
        dest = OUT / "test" / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files[:n_test]:
            shutil.move(str(f), dest / f.name)


def summary():
    print("\nImages per class")
    print(f"{'class':36s}{'train':>8s}{'val':>8s}{'test':>8s}")
    classes = sorted(p.name for p in (OUT / "train").iterdir())
    for c in classes:
        counts = [len(list((OUT / s / c).glob("*"))) if (OUT / s / c).exists() else 0
                  for s in ("train", "val", "test")]
        print(f"{c:36s}{counts[0]:8d}{counts[1]:8d}{counts[2]:8d}")


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists. Delete it first so files are not duplicated.")

    print("PlantVillage")
    copy_split(PV_ROOT / "train", pv_class_name, "pv", "train")
    copy_split(PV_ROOT / "val", pv_class_name, "pv", "val")

    print("\nAgriNet")
    copy_split(AG_ROOT / "train", ag_class_name, "ag", "train")
    copy_split(AG_ROOT / "val", ag_class_name, "ag", "val")

    make_test_split()
    summary()
