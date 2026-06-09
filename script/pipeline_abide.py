"""
pipeline_abide.py
-----------------
Pipeline reproducible para ABIDE/fMRI usando atlas volumetrico configurable:

1. Carga multiples sujetos desde NIfTI ABIDE PCP *_func_preproc.nii.gz.
2. Une cada sujeto con su etiqueta fenotipica DX_GROUP (ASD=1, Control=0).
3. Parcela localmente con atlas, guarda trazabilidad y senales ROI.
4. Estandariza senales ROI.
5. Calcula conectividad para el analisis principal:
   Pearson, Graphical Lasso y LiNGAM.
6. Vectoriza matrices para aprendizaje supervisado.
7. Clasifica ASD vs Control con SVM y Random Forest.
8. Guarda matrices, visor de conectividad, curvas ROC, predicciones por sujeto
   y tabla comparativa CSV.

Uso desde PowerShell, dentro de Pipeline_manual/notebooks:

    python script/pipeline_abide.py `
      --source nifti_atlas `
      --fmri-dir data/ABIDE_pcp/cpac/filt_noglobal `
      --phenotypic data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv `
      --output-dir resultados/pipeline_schaefer_100_all_valid_100rois_tp146 `
      --atlas-name schaefer_100 `
      --methods pearson graphical_lasso lingam `
      --classifiers svm rf `
      --all-available `
      --max-rois 100 `
      --min-timepoints 146 `
      --maxlag 1 `
      --tau-max 1 `
      --skip-bandpass

Metodos implementados pero excluidos del analisis principal actual:

    python script/pipeline_abide.py --methods partial granger pcmci --max-rois 30

Notas:
- Este proyecto no usa ROIs precomputadas de ABIDE PCP.
- Partial, Granger y PCMCI quedan disponibles para pruebas exploratorias, pero no forman
  parte del analisis principal actual.
- Granger, PCMCI y LiNGAM escalan mal con muchas ROIs. Use --max-rois para pruebas.
- PCMCI requiere tigramite instalado.
- LiNGAM requiere lingam instalado.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from .filtrado import filtrar_rois, get_tr
    from .conectividad import (
        correlacion,
        correlacion_parcial,
        graphical_lasso,
        granger,
        lingam,
        pcmci,
        vectorizar,
    )
    from .filtrado import zscore_rois
    from .parcelacion import ATLAS_DEFAULT, ATLAS_DISPONIBLES, cargar_atlas, extraer_senales_roi
except ImportError:
    if __package__:
        raise
    from filtrado import filtrar_rois, get_tr
    from conectividad import (
        correlacion,
        correlacion_parcial,
        graphical_lasso,
        granger,
        lingam,
        pcmci,
        vectorizar,
    )
    from filtrado import zscore_rois
    from parcelacion import ATLAS_DEFAULT, ATLAS_DISPONIBLES, cargar_atlas, extraer_senales_roi


METHODS = {
    "pearson": {"symmetric": True, "label": "Pearson"},
    "partial": {"symmetric": True, "label": "Correlacion parcial"},
    "graphical_lasso": {"symmetric": True, "label": "Graphical Lasso"},
    "granger": {"symmetric": False, "label": "Granger"},
    "pcmci": {"symmetric": False, "label": "PCMCI"},
    "lingam": {"symmetric": False, "label": "DirectLiNGAM"},
}

CLASSIFIERS = {
    "svm": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42),
    "rf": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    ),
}

N_PER_GROUP = 100
CONNECTIVITY_ALGORITHM_VERSION = 6


def enforce_supported_atlas(atlas_name: str) -> str:
    """Valida que el atlas solicitado este soportado por el proyecto."""
    if atlas_name not in ATLAS_DISPONIBLES:
        raise ValueError(
            f"Atlas no permitido: {atlas_name}. Use uno de: {ATLAS_DISPONIBLES}."
        )
    return atlas_name


def diagnosis_name(label: int) -> str:
    return "ASD" if int(label) == 1 else "Control"


def limpiar_numericos(array: np.ndarray, nombre: str = "array") -> np.ndarray:
    """
    Reemplaza NaN/inf por valores finitos para evitar warnings numericos.

    En fMRI puede ocurrir por ROIs constantes, voxeles sin senal o metodos de
    conectividad mal condicionados. Se reemplaza por 0 para no introducir una
    conexion artificial extrema.
    """
    arr = np.asarray(array, dtype=np.float32)
    if not np.isfinite(arr).all():
        n_bad = int(np.size(arr) - np.isfinite(arr).sum())
        print(f"[AVISO] {nombre}: {n_bad} valores NaN/inf reemplazados por 0.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


@dataclass
class Subject:
    subject_id: str
    file_id: str
    label: int
    site: str
    roi_signals_z: np.ndarray
    source_path: Path
    status: str = "ok"
    error_note: str = ""
    tr: Optional[float] = None
    roi_signals_orig: Optional[np.ndarray] = None
    roi_signals_filt: Optional[np.ndarray] = None
    selected_rois: Optional[List[int]] = None
    roi_names: Optional[List[str]] = None
    atlas_data_path: Optional[Path] = None
    fmri_path: Optional[Path] = None
    filter_stage: str = ""
    atlas_name: str = ATLAS_DEFAULT


def parse_file_id(path: Path) -> str:
    """Extrae FILE_ID estilo ABIDE desde nombres de archivos ABIDE PCP."""
    return path.stem


def subject_id_candidates(file_id: str) -> List[str]:
    """Genera candidatos para cruzar nombres de archivo con SUB_ID/FILE_ID."""
    candidates = [file_id]
    numbers = re.findall(r"\d+", file_id)
    if numbers:
        raw = numbers[-1]
        candidates.extend([raw, str(int(raw))])
    return list(dict.fromkeys(candidates))


def load_phenotypic(phenotypic_csv: Path) -> pd.DataFrame:
    """Carga fenotipico ABIDE y normaliza etiqueta: ASD=1, Control=0."""
    df = pd.read_csv(phenotypic_csv)
    required = {"SUB_ID", "FILE_ID", "DX_GROUP"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en fenotipico: {sorted(missing)}")

    df = df.copy()
    df["SUB_ID_STR"] = df["SUB_ID"].astype(str)
    df["FILE_ID_STR"] = df["FILE_ID"].astype(str)
    site_col = "SITE_ID" if "SITE_ID" in df.columns else None
    df["site"] = df[site_col].astype(str) if site_col else ""
    df["label"] = df["DX_GROUP"].map({1: 1, 2: 0})
    df = df[df["label"].isin([0, 1])]
    return df


def find_label(file_id: str, phenotypic: pd.DataFrame) -> Optional[Tuple[str, int, str]]:
    """Busca etiqueta por FILE_ID y, si no aparece, por SUB_ID."""
    candidates = subject_id_candidates(file_id)

    hit = phenotypic[phenotypic["FILE_ID_STR"].isin(candidates)]
    if hit.empty:
        hit = phenotypic[phenotypic["SUB_ID_STR"].isin(candidates)]
    if hit.empty:
        return None

    row = hit.iloc[0]
    return str(row["SUB_ID"]), int(row["label"]), str(row.get("site", ""))


def resolve_class_limits(
    max_subjects: Optional[int],
    n_per_group: Optional[int],
    all_balanced: bool = False,
) -> Dict[int, Optional[int]]:
    """Devuelve limites por clase: label 1=ASD, 0=Control."""
    if all_balanced:
        return {0: None, 1: None}
    if n_per_group is not None:
        return {0: n_per_group, 1: n_per_group}
    if max_subjects is not None:
        return {0: max_subjects // 2, 1: max_subjects - (max_subjects // 2)}
    return {0: None, 1: None}


def enforce_final_balance(subjects: List[Subject], all_balanced: bool) -> List[Subject]:
    """Para modo todos balanceados, recorta a min(ASD, Control)."""
    if not all_balanced:
        return subjects

    by_label = {
        0: [s for s in subjects if s.label == 0],
        1: [s for s in subjects if s.label == 1],
    }
    n = min(len(by_label[0]), len(by_label[1]))
    balanced = sorted(by_label[0][:n] + by_label[1][:n], key=lambda s: s.file_id)
    print(f"Modo todos balanceados: usando {n} ASD y {n} Control.")
    return balanced


def load_roi_1d(path: Path, max_rois: Optional[int] = None) -> np.ndarray:
    """Deshabilitado: el proyecto no usa ROIs precomputadas."""
    raise RuntimeError(
        "Flujo ROI precomputado deshabilitado. Use source='nifti_atlas' con atlas local."
    )
    signals = limpiar_numericos(np.loadtxt(path), nombre=f"senales ROI {path.name}")
    if signals.ndim != 2:
        raise ValueError(f"{path} no tiene una matriz 2D valida")
    if max_rois is not None:
        signals = signals[:, :max_rois]
    return limpiar_numericos(zscore_rois(signals), nombre=f"zscore {path.name}")


def load_subjects_from_roi_files(
    roi_dir: Path,
    phenotypic_csv: Path,
    pattern: str = "",
    max_subjects: Optional[int] = None,
    n_per_group: Optional[int] = N_PER_GROUP,
    all_balanced: bool = False,
    max_rois: Optional[int] = None,
) -> List[Subject]:
    """Deshabilitado: el proyecto no usa ROIs precomputadas."""
    raise RuntimeError(
        "Flujo ROI precomputado deshabilitado. Use source='nifti_atlas' con atlas local."
    )
    phenotypic = load_phenotypic(phenotypic_csv)
    files = sorted(roi_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos {pattern} en {roi_dir}")

    subjects: List[Subject] = []
    skipped: List[str] = []
    seen_file_ids = set()
    seen_subject_ids = set()
    class_counts = {0: 0, 1: 0}
    class_limits = resolve_class_limits(max_subjects, n_per_group, all_balanced)

    for path in files:
        file_id = parse_file_id(path)
        if file_id in seen_file_ids:
            skipped.append(f"{path.name} (duplicado FILE_ID)")
            continue

        label_info = find_label(file_id, phenotypic)
        if label_info is None:
            skipped.append(path.name)
            continue

        subject_id, label, site = label_info
        if subject_id in seen_subject_ids:
            skipped.append(f"{path.name} (duplicado SUB_ID={subject_id})")
            continue
        if class_limits[label] is not None and class_counts[label] >= class_limits[label]:
            continue

        try:
            signals_z = load_roi_1d(path, max_rois=max_rois)
        except Exception as exc:
            skipped.append(f"{path.name} (error lectura ROI: {exc})")
            continue

        roi_names = [f"ROI_{i + 1:03d}" for i in range(signals_z.shape[1])]
        subjects.append(
            Subject(
                subject_id=subject_id,
                file_id=file_id,
                label=label,
                site=site,
                roi_signals_z=signals_z,
                source_path=path,
                roi_names=roi_names,
            )
        )
        seen_file_ids.add(file_id)
        seen_subject_ids.add(subject_id)
        class_counts[label] += 1

        if all(v is not None and class_counts[k] >= v for k, v in class_limits.items()):
            break

    subjects = enforce_final_balance(subjects, all_balanced)
    if skipped:
        print(f"[AVISO] {len(skipped)} archivos omitidos por label faltante, duplicado o error de lectura.")
    if not subjects:
        raise ValueError("No quedo ningun sujeto con etiqueta valida.")

    labels = np.array([s.label for s in subjects])
    print(
        f"Sujetos cargados: {len(subjects)} | ASD={np.sum(labels == 1)} "
        f"| Control={np.sum(labels == 0)} | ROIs={subjects[0].roi_signals_z.shape[1]}"
    )
    return subjects


def parse_func_file_id(path: Path) -> str:
    """Extrae FILE_ID desde ABIDE func_preproc: Pitt_0050003_func_preproc.nii.gz."""
    name = path.name
    for suffix in ["_func_preproc.nii.gz", "_func_preproc.nii", ".nii.gz", ".nii"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_subjects_from_cached_roi_signals(
    output_dir: Path,
    phenotypic_csv: Path,
    atlas_name: str = ATLAS_DEFAULT,
    max_subjects: Optional[int] = None,
    n_per_group: Optional[int] = N_PER_GROUP,
    all_balanced: bool = False,
    max_rois: Optional[int] = None,
) -> List[Subject]:
    """
    Carga sujetos desde senales ROI ya parceladas/filtradas.

    Reutiliza:
    - filtrado_zscore/<file_id>_roi_signals_z.npy
    - parcelacion_atlas/<file_id>/roi_metadata.json

    Esto permite saltarse la etapa cara de abrir NIfTI, remuestrear atlas y
    extraer senales ROI cuando la configuracion de parcelacion no cambio.
    """
    atlas_name = enforce_supported_atlas(atlas_name)
    output_dir = Path(output_dir)
    filt_dir = output_dir / "filtrado_zscore"
    parcel_dir = output_dir / "parcelacion_atlas"
    if not filt_dir.exists():
        raise FileNotFoundError(f"No existe cache de senales ROI: {filt_dir}")

    phenotypic = load_phenotypic(phenotypic_csv)
    files = sorted(filt_dir.glob("*_roi_signals_z.npy"))
    if not files:
        raise FileNotFoundError(f"No se encontraron *_roi_signals_z.npy en {filt_dir}")

    subjects: List[Subject] = []
    skipped: List[str] = []
    seen_file_ids = set()
    seen_subject_ids = set()
    class_counts = {0: 0, 1: 0}
    class_limits = resolve_class_limits(max_subjects, n_per_group, all_balanced)

    for signal_path in files:
        file_id = signal_path.name[: -len("_roi_signals_z.npy")]
        if file_id in seen_file_ids:
            skipped.append(f"{signal_path.name} (duplicado FILE_ID)")
            continue

        label_info = find_label(file_id, phenotypic)
        if label_info is None:
            skipped.append(f"{signal_path.name} (label faltante)")
            continue

        subject_id, label, site = label_info
        if subject_id in seen_subject_ids:
            skipped.append(f"{signal_path.name} (duplicado SUB_ID={subject_id})")
            continue
        if class_limits[label] is not None and class_counts[label] >= class_limits[label]:
            continue

        metadata_path = parcel_dir / file_id / "roi_metadata.json"
        metadata: Dict[str, object] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                skipped.append(f"{signal_path.name} (metadata invalida: {exc})")
                continue

        if metadata and metadata.get("atlas_name", atlas_name) != atlas_name:
            skipped.append(f"{signal_path.name} (atlas cache distinto: {metadata.get('atlas_name')})")
            continue

        try:
            roi_signals_z = limpiar_numericos(np.load(signal_path), nombre=f"cache ROI {file_id}")
        except Exception as exc:
            skipped.append(f"{signal_path.name} (error lectura cache: {exc})")
            continue
        if roi_signals_z.ndim != 2:
            skipped.append(f"{signal_path.name} (senales ROI no son 2D)")
            continue

        selected_rois = metadata.get("selected_rois") if metadata else None
        roi_names = metadata.get("roi_names") if metadata else None
        roi_sizes = metadata.get("roi_sizes") if metadata else None
        if max_rois is not None:
            roi_signals_z = roi_signals_z[:, :max_rois]
            if selected_rois is not None:
                selected_rois = list(selected_rois)[:max_rois]
            if roi_names is not None:
                roi_names = list(roi_names)[:max_rois]
            if roi_sizes is not None:
                roi_sizes = list(roi_sizes)[:max_rois]

        if roi_names is None:
            roi_names = [f"ROI_{i + 1:03d}" for i in range(roi_signals_z.shape[1])]
        if selected_rois is None:
            selected_rois = list(range(1, roi_signals_z.shape[1] + 1))

        atlas_data_path = metadata.get("atlas_data_path") if metadata else None
        fmri_path = metadata.get("fmri_path") if metadata else None
        atlas_data_path = Path(atlas_data_path) if atlas_data_path else parcel_dir / file_id / "atlas_resampled.npy"
        fmri_path = Path(fmri_path) if fmri_path else None

        subjects.append(
            Subject(
                subject_id=subject_id,
                file_id=file_id,
                label=label,
                site=site,
                roi_signals_z=roi_signals_z,
                source_path=signal_path,
                tr=metadata.get("tr") if metadata else None,
                selected_rois=list(selected_rois),
                roi_names=list(roi_names),
                atlas_data_path=atlas_data_path if atlas_data_path.exists() else None,
                fmri_path=fmri_path if fmri_path and fmri_path.exists() else None,
                filter_stage=str(metadata.get("filter_stage", "cache filtrado_zscore")),
                atlas_name=atlas_name,
            )
        )
        seen_file_ids.add(file_id)
        seen_subject_ids.add(subject_id)
        class_counts[label] += 1

        if all(v is not None and class_counts[k] >= v for k, v in class_limits.items()):
            break

    subjects = enforce_final_balance(subjects, all_balanced)
    if skipped:
        print(f"[AVISO] {len(skipped)} caches omitidos por label faltante, duplicado o error.")
    if not subjects:
        raise ValueError("No quedo ningun sujeto cacheado con etiqueta valida.")

    labels = np.array([s.label for s in subjects])
    print(
        f"Sujetos cacheados cargados: {len(subjects)} | ASD={np.sum(labels == 1)} "
        f"| Control={np.sum(labels == 0)} | ROIs={subjects[0].roi_signals_z.shape[1]}"
    )
    return subjects


def load_subjects_from_nifti_atlas(
    fmri_dir: Path,
    phenotypic_csv: Path,
    output_dir: Path,
    atlas_name: str = ATLAS_DEFAULT,
    pattern: str = "*_func_preproc.nii.gz",
    max_subjects: Optional[int] = None,
    n_per_group: Optional[int] = N_PER_GROUP,
    all_balanced: bool = False,
    max_rois: Optional[int] = None,
    min_voxels: int = 100,
    lowcut: float = 0.01,
    highcut: float = 0.1,
    apply_bandpass: bool = True,
) -> List[Subject]:
    """
    Carga sujetos desde NIfTI 4D, aplica parcelacion con atlas y guarda etapas.

    Este es el modo alineado con el flujo:
    ABIDE -> preprocesamiento -> atlas -> extraccion ROI -> filtrado -> z-score.
    """
    atlas_name = enforce_supported_atlas(atlas_name)
    path_parts = {part.lower() for part in fmri_dir.parts}
    if apply_bandpass and any(part.startswith("filt_") for part in path_parts):
        raise ValueError(
            "Los NIfTI parecen venir de una carpeta ya filtrada de ABIDE PCP "
            f"({fmri_dir}). Usa --skip-bandpass/apply_bandpass=False para evitar doble filtrado, "
            "o descarga func_preproc sin bandpass."
        )
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Este modo requiere nibabel. Instala con: pip install nibabel") from exc

    phenotypic = load_phenotypic(phenotypic_csv)
    files = sorted(fmri_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos {pattern} en {fmri_dir}")

    parcel_dir = output_dir / "parcelacion_atlas"
    filt_dir = output_dir / "filtrado_zscore"
    parcel_dir.mkdir(parents=True, exist_ok=True)
    filt_dir.mkdir(parents=True, exist_ok=True)

    subjects: List[Subject] = []
    skipped: List[str] = []
    seen_file_ids = set()
    seen_subject_ids = set()
    class_counts = {0: 0, 1: 0}
    class_limits = resolve_class_limits(max_subjects, n_per_group, all_balanced)

    for idx, path in enumerate(files, start=1):
        file_id = parse_func_file_id(path)
        if file_id in seen_file_ids:
            skipped.append(f"{path.name} (duplicado FILE_ID)")
            continue

        label_info = find_label(file_id, phenotypic)
        if label_info is None:
            skipped.append(path.name)
            continue

        subject_id, label, site = label_info
        if subject_id in seen_subject_ids:
            skipped.append(f"{path.name} (duplicado SUB_ID={subject_id})")
            continue
        if class_limits[label] is not None and class_counts[label] >= class_limits[label]:
            continue

        subject_dir = parcel_dir / file_id
        atlas_path = subject_dir / "atlas_resampled.npy"
        signal_cache_path = filt_dir / f"{file_id}_roi_signals_z.npy"
        metadata_path = subject_dir / "roi_metadata.json"
        roi_signals = None
        roi_signals_filt = None
        roi_signals_z = None
        selected_rois = None
        roi_names = None
        tr = None
        filter_stage = ""
        used_roi_cache = False

        if signal_cache_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("atlas_name") == atlas_name:
                    roi_signals_z = limpiar_numericos(np.load(signal_cache_path), nombre=f"cache ROI {file_id}")
                    selected_rois = list(metadata.get("selected_rois") or [])
                    roi_names = list(metadata.get("roi_names") or [])
                    roi_sizes = metadata.get("roi_sizes") or {}
                    if max_rois is not None:
                        if roi_signals_z.shape[1] < max_rois:
                            raise ValueError(
                                f"cache con {roi_signals_z.shape[1]} ROIs no alcanza max_rois={max_rois}"
                            )
                        roi_signals_z = roi_signals_z[:, :max_rois]
                        selected_rois = selected_rois[:max_rois]
                        roi_names = roi_names[:max_rois]
                    if not selected_rois:
                        selected_rois = list(range(1, roi_signals_z.shape[1] + 1))
                    if not roi_names:
                        roi_names = [f"ROI_{i + 1:03d}" for i in range(roi_signals_z.shape[1])]
                    tr = metadata.get("tr")
                    atlas_data_path = Path(metadata.get("atlas_data_path", atlas_path))
                    atlas_path = atlas_data_path if atlas_data_path.exists() else atlas_path
                    filter_stage = str(metadata.get("filter_stage", "cache filtrado_zscore"))
                    used_roi_cache = True
                    print(f"\nCache ROI {idx}/{len(files)} {file_id}: cargado sin re-parcelar.")
            except Exception as exc:
                print(f"[AVISO] Cache ROI incompatible para {file_id}: {exc}. Recalculando.")

        if not used_roi_cache:
            try:
                print(f"\nParcelando {idx}/{len(files)} {file_id} con atlas {atlas_name}...")
                fmri_img = nib.load(str(path))
                fmri_data = fmri_img.get_fdata(dtype=np.float32)
                tr = get_tr(fmri_img)

                atlas_data, atlas_obj = cargar_atlas(fmri_img, atlas_name=atlas_name)
                roi_signals, selected_rois, roi_names, roi_sizes = extraer_senales_roi(
                    fmri_data,
                    atlas_data,
                    atlas_obj,
                    min_voxels=min_voxels,
                )
                if max_rois is not None:
                    roi_signals = roi_signals[:, :max_rois]
                    selected_rois = selected_rois[:max_rois]
                    roi_names = roi_names[:max_rois]

                roi_signals = limpiar_numericos(roi_signals, nombre=f"roi_signals {file_id}")
                if apply_bandpass:
                    roi_signals_filt = limpiar_numericos(
                        filtrar_rois(roi_signals, tr=tr, lowcut=lowcut, highcut=highcut),
                        nombre=f"roi_signals_filt {file_id}",
                    )
                    filter_stage = "bandpass local + zscore local"
                else:
                    roi_signals_filt = roi_signals.copy()
                    filter_stage = "sin bandpass local + zscore local"
                roi_signals_z = limpiar_numericos(zscore_rois(roi_signals_filt), nombre=f"roi_signals_z {file_id}")
            except Exception as exc:
                skipped.append(f"{path.name} (error parcelacion/filtrado: {exc})")
                continue

            subject_dir.mkdir(parents=True, exist_ok=True)
            np.save(atlas_path, atlas_data.astype(np.int16))
            np.save(subject_dir / "roi_signals_orig.npy", roi_signals)
            np.save(subject_dir / "roi_signals_filt.npy", roi_signals_filt)
            np.save(signal_cache_path, roi_signals_z)
            save_json(subject_dir / "roi_metadata.json", {
                "file_id": file_id,
                "subject_id": subject_id,
                "diagnosis": diagnosis_name(label),
                "site": site,
                "atlas_name": atlas_name,
                "tr": tr,
                "selected_rois": selected_rois,
                "roi_names": roi_names,
                "roi_sizes": roi_sizes,
                "fmri_path": str(path),
                "atlas_data_path": str(atlas_path),
                "filter_stage": filter_stage,
                "max_rois": max_rois,
            })

        subjects.append(
            Subject(
                subject_id=subject_id,
                file_id=file_id,
                label=label,
                site=site,
                roi_signals_z=roi_signals_z,
                source_path=path,
                tr=tr,
                roi_signals_orig=roi_signals,
                roi_signals_filt=roi_signals_filt,
                selected_rois=selected_rois,
                roi_names=roi_names,
                atlas_data_path=atlas_path if atlas_path.exists() else None,
                fmri_path=path,
                filter_stage=filter_stage,
                atlas_name=atlas_name,
            )
        )
        seen_file_ids.add(file_id)
        seen_subject_ids.add(subject_id)
        class_counts[label] += 1

        if all(v is not None and class_counts[k] >= v for k, v in class_limits.items()):
            break

    subjects = enforce_final_balance(subjects, all_balanced)
    if skipped:
        print(f"[AVISO] {len(skipped)} NIfTI omitidos por label faltante, duplicado o error.")
    if not subjects:
        preview = "\n".join(f"- {item}" for item in skipped[:15])
        raise ValueError(
            "No quedo ningun sujeto NIfTI con etiqueta valida. "
            f"Archivos encontrados={len(files)} | omitidos={len(skipped)}. "
            f"Primeros motivos:\n{preview}"
        )

    labels = np.array([s.label for s in subjects])
    print(
        f"Sujetos atlas cargados: {len(subjects)} | ASD={np.sum(labels == 1)} "
        f"| Control={np.sum(labels == 0)} | ROIs={subjects[0].roi_signals_z.shape[1]}"
    )
    return subjects


def compute_matrix(
    signals_z: np.ndarray,
    method: str,
    maxlag: int,
    tau_max: int,
    lingam_random_state: int = 42,
    granger_lag_strategy: str = "min_q",
    pcmci_value_mode: str = "signed_logp",
) -> np.ndarray:
    """Calcula una matriz de conectividad para un sujeto."""
    signals_z = limpiar_numericos(signals_z, nombre=f"entrada conectividad {method}")
    if method == "pearson":
        return limpiar_numericos(correlacion(signals_z), nombre="matriz pearson")
    if method == "partial":
        return limpiar_numericos(correlacion_parcial(signals_z), nombre="matriz partial")
    if method == "graphical_lasso":
        matrix, _ = graphical_lasso(signals_z)
        return limpiar_numericos(matrix, nombre="matriz graphical_lasso")
    if method == "granger":
        matrix, _ = granger(signals_z, maxlag=maxlag, lag_strategy=granger_lag_strategy)
        return limpiar_numericos(matrix, nombre="matriz granger")
    if method == "pcmci":
        matrix, _, _ = pcmci(signals_z, tau_max=tau_max, value_mode=pcmci_value_mode)
        return limpiar_numericos(matrix, nombre="matriz pcmci")
    if method == "lingam":
        matrix, _, _ = lingam(signals_z, random_state=lingam_random_state)
        return limpiar_numericos(matrix, nombre="matriz lingam")
    raise ValueError(f"Metodo no reconocido: {method}")


def validate_subjects(subjects: Sequence[Subject], atlas_name: str) -> None:
    """Valida atlas, shape de senales y orden comun de ROIs."""
    atlas_name = enforce_supported_atlas(atlas_name)
    if not subjects:
        raise ValueError("No hay sujetos para validar.")

    first = subjects[0]
    expected_n_rois = first.roi_signals_z.shape[1]
    expected_selected = tuple(first.selected_rois or [])
    expected_names = tuple(first.roi_names or [])

    for subject in subjects:
        if subject.atlas_name != atlas_name:
            raise ValueError(f"{subject.file_id}: atlas inesperado {subject.atlas_name}.")
        if subject.roi_signals_z.ndim != 2:
            raise ValueError(f"{subject.file_id}: senales ROI no son 2D.")
        if subject.roi_signals_z.shape[1] != expected_n_rois:
            raise ValueError(f"{subject.file_id}: n_rois inconsistente.")
        if not np.isfinite(subject.roi_signals_z).all():
            raise ValueError(f"{subject.file_id}: senales ROI contienen NaN/inf.")
        if expected_selected and tuple(subject.selected_rois or []) != expected_selected:
            raise ValueError(f"{subject.file_id}: orden/IDs de ROIs no coincide.")
        if expected_names and tuple(subject.roi_names or []) != expected_names:
            raise ValueError(f"{subject.file_id}: nombres de ROIs no coinciden.")

    labels = np.array([s.label for s in subjects])
    print(
        f"Validacion sujetos: atlas={atlas_name} | sujetos={len(subjects)} | "
        f"ASD={np.sum(labels == 1)} | Control={np.sum(labels == 0)} | "
        f"ROIs={expected_n_rois} | T primer sujeto={first.roi_signals_z.shape[0]}"
    )


def build_timepoint_site_table(
    subjects: Sequence[Subject],
    min_timepoints: Optional[int] = None,
    include_total: bool = True,
) -> pd.DataFrame:
    """Resume por sitio si los sujetos tienen suficientes volumenes temporales."""
    if not subjects:
        return pd.DataFrame()

    subject_rows = []
    for subject in subjects:
        n_timepoints, n_rois = subject.roi_signals_z.shape
        subject_rows.append(
            {
                "site": subject.site,
                "diagnosis": diagnosis_name(subject.label),
                "n_timepoints": int(n_timepoints),
                "n_rois": int(n_rois),
                "timepoints_minus_rois": int(n_timepoints - n_rois),
                "timepoints_per_roi": float(n_timepoints / n_rois) if n_rois else np.nan,
            }
        )

    subject_df = pd.DataFrame(subject_rows)

    def _summarize(group: pd.DataFrame, site_label: str) -> Dict[str, object]:
        row: Dict[str, object] = {
            "site": site_label,
            "n_subjects": int(len(group)),
            "asd": int(np.sum(group["diagnosis"] == "ASD")),
            "control": int(np.sum(group["diagnosis"] == "Control")),
            "n_rois_min": int(group["n_rois"].min()),
            "n_rois_max": int(group["n_rois"].max()),
            "timepoints_min": int(group["n_timepoints"].min()),
            "timepoints_mean": float(group["n_timepoints"].mean()),
            "timepoints_median": float(group["n_timepoints"].median()),
            "timepoints_max": int(group["n_timepoints"].max()),
            "timepoints_per_roi_min": float(group["timepoints_per_roi"].min()),
            "n_t_le_rois": int(np.sum(group["n_timepoints"] <= group["n_rois"])),
            "n_t_lt_2x_rois": int(np.sum(group["n_timepoints"] < 2 * group["n_rois"])),
        }
        if min_timepoints is not None:
            row["min_timepoints_threshold"] = int(min_timepoints)
            row["n_below_threshold"] = int(np.sum(group["n_timepoints"] < min_timepoints))
        return row

    rows = [_summarize(group, str(site)) for site, group in subject_df.groupby("site", sort=True)]
    if include_total:
        rows.append(_summarize(subject_df, "TOTAL"))
    return pd.DataFrame(rows)


def save_timepoint_site_table(
    subjects: Sequence[Subject],
    output_dir: Path,
    min_timepoints: Optional[int] = None,
) -> Optional[Path]:
    """Guarda tabla por sitio con distribucion de timepoints."""
    table = build_timepoint_site_table(subjects, min_timepoints=min_timepoints)
    if table.empty:
        return None
    path = output_dir / "resumen_timepoints_por_sitio.csv"
    table.to_csv(path, index=False)
    return path


def filter_subjects_by_timepoints(subjects: Sequence[Subject], min_timepoints: int) -> List[Subject]:
    """Filtra sujetos con menos volumenes temporales que el umbral indicado."""
    filtered = [subject for subject in subjects if subject.roi_signals_z.shape[0] >= min_timepoints]
    if not filtered:
        raise ValueError(f"El filtro min_timepoints={min_timepoints} elimino todos los sujetos.")
    return filtered


def validate_connectivity_matrix(
    matrix: np.ndarray,
    method: str,
    expected_shape: Tuple[int, int],
    subject_id: str,
) -> None:
    """Valida shape, finitud, diagonal y simetria/direccion segun metodo."""
    if matrix.shape != expected_shape:
        raise ValueError(f"{subject_id} {method}: shape {matrix.shape} != {expected_shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{subject_id} {method}: matriz contiene NaN/inf.")
    symmetric = bool(METHODS[method]["symmetric"])
    if symmetric and not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{subject_id} {method}: se esperaba matriz simetrica.")
    if not symmetric:
        diag_abs = float(np.max(np.abs(np.diag(matrix))))
        if diag_abs > 1e-6:
            raise ValueError(f"{subject_id} {method}: diagonal causal no es cero (max={diag_abs:.3g}).")
        if np.allclose(matrix, matrix.T, atol=1e-6):
            print(f"[AVISO] {subject_id} {method}: matriz dirigida casi simetrica; revisar interpretacion.")


def enforce_symmetric_connectivity_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    """Corrige asimetrias numericas en metodos que por definicion son simetricos."""
    matrix = limpiar_numericos(np.asarray(matrix, dtype=float), nombre=f"matriz {method}")
    if METHODS[method]["symmetric"]:
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)
    return matrix


def save_feature_map(output_dir: Path, method: str, roi_names: Sequence[str]) -> Path:
    """Guarda el orden exacto de features para trazabilidad."""
    n_rois = len(roi_names)
    symmetric = bool(METHODS[method]["symmetric"])
    rows = []
    if symmetric:
        pairs = ((i, j) for i in range(n_rois) for j in range(i + 1, n_rois))
    else:
        pairs = ((i, j) for i in range(n_rois) for j in range(n_rois) if i != j)
    for feature_index, (i, j) in enumerate(pairs):
        rows.append(
            {
                "feature_index": feature_index,
                "method": method,
                "directed": not symmetric,
                "roi_origen_idx": i,
                "roi_destino_idx": j,
                "roi_origen": roi_names[i],
                "roi_destino": roi_names[j],
            }
        )
    feature_dir = output_dir / "feature_maps"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{method}_feature_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def validate_feature_dataset(X: np.ndarray, y: np.ndarray, method: str, n_rois: int) -> None:
    """Valida dimension de features y distribucion de clases."""
    expected = n_rois * (n_rois - 1) // 2 if METHODS[method]["symmetric"] else n_rois * (n_rois - 1)
    if X.shape[1] != expected:
        raise ValueError(f"{method}: n_features {X.shape[1]} != esperado {expected}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{method}: X contiene NaN/inf.")
    counts = np.bincount(y, minlength=2)
    print(
        f"Validacion features {method}: X={X.shape} | ASD={counts[1]} | "
        f"Control={counts[0]} | simetrica={METHODS[method]['symmetric']}"
    )


def matrix_cache_signature(
    subject: Subject,
    method: str,
    maxlag: int,
    tau_max: int,
    lingam_random_state: int,
    granger_lag_strategy: str,
    pcmci_value_mode: str,
) -> Dict[str, object]:
    """Firma para evitar reusar matrices de otro atlas, ROI set o parametro."""
    return {
        "connectivity_algorithm_version": CONNECTIVITY_ALGORITHM_VERSION,
        "file_id": subject.file_id,
        "atlas_name": subject.atlas_name,
        "selected_rois": list(subject.selected_rois or []),
        "roi_names": list(subject.roi_names or []),
        "filter_stage": subject.filter_stage,
        "method": method,
        "n_rois": int(subject.roi_signals_z.shape[1]),
        "n_timepoints": int(subject.roi_signals_z.shape[0]),
        "maxlag": maxlag,
        "tau_max": tau_max,
        "lingam_random_state": lingam_random_state,
        "granger_lag_strategy": granger_lag_strategy,
        "pcmci_value_mode": pcmci_value_mode,
    }


def cache_meta_matches(meta_path: Path, signature: Dict[str, object]) -> bool:
    if not meta_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")) == signature
    except Exception:
        return False


def build_method_dataset(
    subjects: Sequence[Subject],
    method: str,
    output_dir: Path,
    maxlag: int = 2,
    tau_max: int = 2,
    lingam_random_state: int = 42,
    granger_lag_strategy: str = "min_q",
    pcmci_value_mode: str = "signed_logp",
    cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    """Calcula/cachea matrices y devuelve X, y, matrices."""
    if method not in METHODS:
        raise ValueError(f"Metodo {method} no disponible. Use: {list(METHODS)}")

    symmetric = bool(METHODS[method]["symmetric"])
    expected_n_rois = subjects[0].roi_signals_z.shape[1]
    expected_shape = (expected_n_rois, expected_n_rois)
    method_dir = output_dir / "matrices" / f"{method}_{expected_n_rois}rois"
    method_dir.mkdir(parents=True, exist_ok=True)

    matrices = []
    matrix_paths = []
    vectors = []
    y = []

    print(f"\nMetodo: {method}")
    for idx, subject in enumerate(subjects, start=1):
        t0 = time.time()
        matrix_path = method_dir / f"{subject.file_id}_{method}.npy"
        meta_path = matrix_path.with_suffix(".meta.json")
        signature = matrix_cache_signature(
            subject,
            method,
            maxlag=maxlag,
            tau_max=tau_max,
            lingam_random_state=lingam_random_state,
            granger_lag_strategy=granger_lag_strategy,
            pcmci_value_mode=pcmci_value_mode,
        )

        matrix = None
        if cache and matrix_path.exists():
            cached_matrix = np.load(matrix_path)
            if cached_matrix.shape == expected_shape and cache_meta_matches(meta_path, signature):
                matrix = cached_matrix
            else:
                print(
                    f"  Cache incompatible para {subject.file_id}: "
                    f"{cached_matrix.shape} != {expected_shape} o metadata distinta; recalculando."
                )

        if matrix is None:
            matrix = compute_matrix(
                subject.roi_signals_z,
                method,
                maxlag=maxlag,
                tau_max=tau_max,
                lingam_random_state=lingam_random_state,
                granger_lag_strategy=granger_lag_strategy,
                pcmci_value_mode=pcmci_value_mode,
            )
            if cache:
                np.save(matrix_path, matrix)
                save_json(meta_path, signature)

        matrix = enforce_symmetric_connectivity_matrix(matrix, method)
        if cache:
            np.save(matrix_path, matrix)
            save_json(meta_path, signature)

        validate_connectivity_matrix(matrix, method, expected_shape, subject.file_id)

        matrices.append(matrix)
        matrix_paths.append(matrix_path)
        vectors.append(vectorizar(matrix, simetrica=symmetric))
        y.append(subject.label)
        print(
            f"  {idx:03d}/{len(subjects)} {subject.file_id} "
            f"label={subject.label} features={len(vectors[-1])} tiempo={time.time() - t0:.1f}s"
        )

    X = limpiar_numericos(np.asarray(vectors, dtype=np.float32), nombre=f"X_{method}")
    y_arr = np.asarray(y, dtype=int)
    matrix_arr = limpiar_numericos(np.asarray(matrices, dtype=np.float32), nombre=f"matrices_{method}")
    np.save(output_dir / f"X_{method}.npy", X)
    np.save(output_dir / f"y_{method}.npy", y_arr)
    save_feature_map(
        output_dir,
        method,
        subjects[0].roi_names or [f"ROI_{i + 1:03d}" for i in range(expected_n_rois)],
    )
    validate_feature_dataset(X, y_arr, method, expected_n_rois)
    return X, y_arr, matrix_arr, matrix_paths


def matrix_to_edges(matrix: np.ndarray, method: str, roi_names: Sequence[str]) -> pd.DataFrame:
    """Convierte una matriz de conectividad a tabla ROI-origen/ROI-destino."""
    symmetric = bool(METHODS[method]["symmetric"])
    rows = []
    n_rois = matrix.shape[0]

    if len(roi_names) != n_rois:
        roi_names = [f"ROI_{i + 1:03d}" for i in range(n_rois)]

    if symmetric:
        pairs = ((i, j) for i in range(n_rois) for j in range(i + 1, n_rois))
    else:
        pairs = ((i, j) for i in range(n_rois) for j in range(n_rois) if i != j)

    for i, j in pairs:
        value = float(matrix[i, j])
        rows.append(
            {
                "roi_origen": roi_names[i],
                "roi_destino": roi_names[j],
                "valor_conectividad": value,
                "valor_abs": abs(value),
            }
        )

    return pd.DataFrame(rows)


def save_connectivity_edge_table(
    subjects: Sequence[Subject],
    method: str,
    matrices: np.ndarray,
    output_dir: Path,
) -> Path:
    """Guarda tabla completa de conexiones por sujeto y metodo."""
    table_dir = output_dir / "conectividad_tablas"
    table_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    family = "causal/direccional" if not METHODS[method]["symmetric"] else "asociativo"
    for subject, matrix in zip(subjects, matrices):
        roi_names = subject.roi_names or [f"ROI_{i + 1:03d}" for i in range(matrix.shape[0])]
        df = matrix_to_edges(matrix, method, roi_names)
        df.insert(0, "method", method)
        df.insert(1, "connectivity_family", family)
        df.insert(2, "subject_id", subject.subject_id)
        df.insert(3, "file_id", subject.file_id)
        df.insert(4, "diagnosis", diagnosis_name(subject.label))
        df.insert(5, "site", subject.site)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    path = table_dir / f"{method}_edges.csv"
    out.to_csv(path, index=False)
    return path


def summarize_connectivity_matrices(
    subjects: Sequence[Subject],
    method: str,
    matrices: np.ndarray,
) -> List[Dict[str, object]]:
    """Resume densidad, rango y simetria de matrices por sujeto."""
    rows: List[Dict[str, object]] = []
    symmetric = bool(METHODS[method]["symmetric"])
    family = "causal/direccional" if not symmetric else "asociativo"

    for subject, matrix in zip(subjects, matrices):
        matrix = np.asarray(matrix, dtype=float)
        n_rois = matrix.shape[0]
        if symmetric:
            values = matrix[np.triu_indices(n_rois, k=1)]
        else:
            values = matrix[~np.eye(n_rois, dtype=bool)]
        nonzero = np.abs(values) > 1e-12
        rows.append(
            {
                "subject_id": subject.subject_id,
                "file_id": subject.file_id,
                "diagnosis": diagnosis_name(subject.label),
                "site": subject.site,
                "method": method,
                "connectivity_family": family,
                "n_timepoints": subject.roi_signals_z.shape[0],
                "n_rois": n_rois,
                "n_features": int(values.size),
                "n_nonzero": int(np.sum(nonzero)),
                "density": float(np.mean(nonzero)) if values.size else np.nan,
                "mean": float(np.mean(values)) if values.size else np.nan,
                "std": float(np.std(values)) if values.size else np.nan,
                "min": float(np.min(values)) if values.size else np.nan,
                "max": float(np.max(values)) if values.size else np.nan,
                "abs_mean": float(np.mean(np.abs(values))) if values.size else np.nan,
                "abs_p95": float(np.percentile(np.abs(values), 95)) if values.size else np.nan,
                "diag_max_abs": float(np.max(np.abs(np.diag(matrix)))),
                "symmetry_max_abs_diff": float(np.max(np.abs(matrix - matrix.T))),
            }
        )
    return rows


def save_matrix_summary_table(matrix_rows: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    """Guarda resumen de densidad/sparsity de matrices por sujeto y metodo."""
    if not matrix_rows:
        return None
    path = output_dir / "resumen_matrices_conectividad.csv"
    pd.DataFrame(matrix_rows).to_csv(path, index=False)
    return path


def valid_cv_splits(y: np.ndarray, requested: int) -> int:
    """Ajusta folds al minimo por clase."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            "Se necesitan sujetos de ambas clases para clasificar. "
            f"Clases disponibles: {unique_classes.tolist()}."
        )

    counts = np.bincount(y, minlength=2)
    max_splits = int(counts[counts > 0].min())
    if max_splits < 2:
        raise ValueError(
            "Se necesitan al menos 2 sujetos por clase para validacion cruzada. "
            f"Clases disponibles: ASD={counts[1]}, Control={counts[0]}."
        )
    return max(2, min(requested, max_splits))


def build_cv(
    y: np.ndarray,
    n_splits: int,
    cv_strategy: str = "group_site",
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
):
    """Construye validacion cruzada estratificada o estratificada agrupada por sitio."""
    if cv_strategy == "stratified":
        cv_splits = valid_cv_splits(y, n_splits)
        return (
            StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
            cv_splits,
            None,
        )

    if cv_strategy == "group_site":
        if groups is None:
            raise ValueError("cv_strategy='group_site' requiere grupos de sitio.")
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("Se necesitan al menos 2 sitios para validacion agrupada.")
        cv_splits = min(valid_cv_splits(y, n_splits), n_groups)
        return (
            StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
            cv_splits,
            groups,
        )

    raise ValueError("cv_strategy debe ser 'stratified' o 'group_site'")


def evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    n_splits: int,
    cv_strategy: str = "stratified",
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """Evalua clasificador con predicciones out-of-fold y metricas por fold."""
    if classifier_name not in CLASSIFIERS:
        raise ValueError(f"Clasificador no reconocido: {classifier_name}")

    X = limpiar_numericos(X, nombre=f"X clasificador {classifier_name}")
    y = np.asarray(y, dtype=int)
    cv, cv_splits, cv_groups = build_cv(
        y,
        n_splits=n_splits,
        cv_strategy=cv_strategy,
        groups=groups,
        random_state=random_state,
    )
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clone(CLASSIFIERS[classifier_name])),
        ]
    )

    proba = np.full(len(y), np.nan, dtype=float)
    pred = np.full(len(y), -1, dtype=int)
    fold_metrics: List[Dict[str, object]] = []
    split_iter = cv.split(X, y, cv_groups) if cv_groups is not None else cv.split(X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            train_group_msg = ""
            if cv_groups is not None:
                train_group_msg = (
                    " | grupos train="
                    + "|".join(sorted(map(str, np.unique(cv_groups[train_idx]))))
                    + " | grupos test="
                    + "|".join(sorted(map(str, np.unique(cv_groups[test_idx]))))
                )
            raise ValueError(
                f"Fold {fold_idx} tiene una sola clase en entrenamiento "
                f"(ASD={int(np.sum(y_train == 1))}, Control={int(np.sum(y_train == 0))}). "
                "Con muestras pequenas, usa cv_strategy='stratified' o reduce n_splits; "
                "reserva cv_strategy='group_site' para una muestra mas grande."
                + train_group_msg
            )

        fold_estimator = clone(estimator)
        fold_estimator.fit(X[train_idx], y[train_idx])
        raw_proba = fold_estimator.predict_proba(X[test_idx])
        classes = list(fold_estimator.named_steps["clf"].classes_)
        if 1 not in classes:
            raise ValueError(
                f"Fold {fold_idx} entreno sin clase ASD; no se puede calcular probabilidad ASD."
            )
        fold_proba = raw_proba[:, classes.index(1)]
        fold_pred = (fold_proba >= 0.5).astype(int)
        proba[test_idx] = fold_proba
        pred[test_idx] = fold_pred

        y_test = y[test_idx]
        fold_auc = np.nan
        if len(np.unique(y_test)) == 2:
            fold_auc = roc_auc_score(y_test, fold_proba)
        fold_row: Dict[str, object] = {
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_asd": int(np.sum(y[train_idx] == 1)),
            "n_train_control": int(np.sum(y[train_idx] == 0)),
            "n_test_asd": int(np.sum(y_test == 1)),
            "n_test_control": int(np.sum(y_test == 0)),
            "accuracy": accuracy_score(y_test, fold_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, fold_pred),
            "auc": fold_auc,
            "f1": f1_score(y_test, fold_pred, zero_division=0),
            "precision": precision_score(y_test, fold_pred, zero_division=0),
            "recall": recall_score(y_test, fold_pred, zero_division=0),
            "specificity": recall_score(y_test, fold_pred, pos_label=0, zero_division=0),
        }
        if cv_groups is not None:
            fold_row["test_groups"] = "|".join(sorted(map(str, np.unique(cv_groups[test_idx]))))
        fold_metrics.append(fold_row)

    if np.isnan(proba).any() or np.any(pred < 0):
        raise RuntimeError("La validacion cruzada no genero predicciones para todos los sujetos.")

    pred = (proba >= 0.5).astype(int)
    fold_df = pd.DataFrame(fold_metrics)

    metric_summary: Dict[str, float] = {}
    for metric in ["accuracy", "balanced_accuracy", "auc", "f1", "precision", "recall", "specificity"]:
        values = pd.to_numeric(fold_df[metric], errors="coerce").dropna().to_numpy(dtype=float)
        metric_summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
        metric_summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    return {
        "classifier": classifier_name,
        "n_splits": cv_splits,
        "cv_strategy": cv_strategy,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auc": roc_auc_score(y, proba),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        **metric_summary,
        "fold_metrics": fold_metrics,
        "y_true": y,
        "y_score": proba,
        "y_pred": pred,
    }


def plot_mean_matrix(matrix_stack: np.ndarray, method: str, output_dir: Path) -> Path:
    """Guarda heatmap de matriz promedio por metodo."""
    mean_matrix = np.nanmean(matrix_stack, axis=0)
    vmax = np.nanpercentile(np.abs(mean_matrix), 98)
    vmax = float(vmax) if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Matriz promedio - {METHODS[method]['label']}")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = output_dir / f"matriz_promedio_{method}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_subject_matrix(matrix: np.ndarray, subject: Subject, method: str, output_dir: Path) -> Path:
    """Guarda un heatmap individual para inspeccion tipo visor."""
    subject_dir = output_dir / "visor_conectividad" / subject.file_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    vmax = np.nanpercentile(np.abs(matrix), 98)
    vmax = float(vmax) if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    diagnosis = diagnosis_name(subject.label)
    ax.set_title(f"{subject.file_id} - {diagnosis}\n{METHODS[method]['label']}")
    ax.set_xlabel("ROI destino" if not METHODS[method]["symmetric"] else "ROI")
    ax.set_ylabel("ROI origen" if not METHODS[method]["symmetric"] else "ROI")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = subject_dir / f"matriz_{method}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_stage_outputs(subjects: Sequence[Subject], output_dir: Path) -> Dict[str, str]:
    """
    Guarda salidas intermedias por sujeto.

    Guarda senales intermedias de NIfTI parcelado localmente con atlas.
    """
    parcel_dir = output_dir / "parcelacion_atlas"
    filt_dir = output_dir / "filtrado_zscore"
    parcel_dir.mkdir(parents=True, exist_ok=True)
    filt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject in subjects:
        signals_path = filt_dir / f"{subject.file_id}_roi_signals_z.npy"
        np.save(signals_path, subject.roi_signals_z)
        rows.append(
            {
                "subject_id": subject.subject_id,
                "file_id": subject.file_id,
                "diagnosis": diagnosis_name(subject.label),
                "site": subject.site,
                "source": str(subject.source_path),
                "atlas_name": subject.atlas_name,
                "roi_stage": f"atlas {subject.atlas_name} local sobre NIfTI",
                "filter_stage": subject.filter_stage,
                "roi_signals_z": str(signals_path),
                "atlas_data_path": str(subject.atlas_data_path) if subject.atlas_data_path is not None else "",
                "fmri_path": str(subject.fmri_path) if subject.fmri_path is not None else "",
                "n_timepoints": subject.roi_signals_z.shape[0],
                "n_rois": subject.roi_signals_z.shape[1],
            }
        )

    manifest_path = output_dir / "trazabilidad_parcelacion_filtrado.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    atlas_label = subjects[0].atlas_name if subjects else ATLAS_DEFAULT
    readme_path = parcel_dir / "README.txt"
    readme_path.write_text(
        f"Modo actual: NIfTI + atlas {atlas_label}.\n"
        "Cada carpeta de sujeto contiene atlas_resampled.npy, senales ROI "
        "originales/filtradas y roi_metadata.json para abrir el visor del atlas.\n",
        encoding="utf-8",
    )
    return {"manifest": str(manifest_path), "parcelacion": str(parcel_dir), "filtrado": str(filt_dir)}


def plot_roc_curves(results: Sequence[Dict[str, object]], output_dir: Path) -> Path:
    """Guarda curvas ROC para todas las combinaciones metodo x clasificador."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in results:
        y_true = np.asarray(res["y_true"])
        y_score = np.asarray(res["y_score"])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            lw=1.8,
            label=f"{res['method']} + {res['classifier']} (AUC={roc_auc:.2f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Azar")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC - ASD vs Control")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    path = output_dir / "curvas_roc.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def append_subject_predictions(
    prediction_rows: List[Dict[str, object]],
    subjects: Sequence[Subject],
    method: str,
    classifier: str,
    res: Dict[str, object],
) -> None:
    """Agrega predicciones por sujeto para trazabilidad clinica/visual."""
    y_score = np.asarray(res["y_score"])
    y_pred = np.asarray(res["y_pred"])
    y_true = np.asarray(res["y_true"])

    for idx, subject in enumerate(subjects):
        pred_label = int(y_pred[idx])
        true_label = int(y_true[idx])
        prediction_rows.append(
            {
                "subject_id": subject.subject_id,
                "file_id": subject.file_id,
                "true_label": true_label,
                "true_diagnosis": diagnosis_name(true_label),
                "site": subject.site,
                "method": method,
                "connectivity_family": "causal/direccional" if not METHODS[method]["symmetric"] else "asociativo",
                "classifier": classifier,
                "pred_label": pred_label,
                "pred_diagnosis": diagnosis_name(pred_label),
                "prob_asd": float(y_score[idx]),
                "correct": bool(pred_label == true_label),
            }
        )


def append_fold_metric_rows(
    fold_rows: List[Dict[str, object]],
    method: str,
    classifier: str,
    res: Dict[str, object],
) -> None:
    """Agrega metricas por fold para reportar estabilidad de la validacion."""
    for row in res.get("fold_metrics", []):
        fold_rows.append(
            {
                "method": method,
                "connectivity_family": "causal/direccional" if not METHODS[method]["symmetric"] else "asociativo",
                "classifier": classifier,
                "cv_strategy": res.get("cv_strategy", ""),
                **row,
            }
        )


def save_prediction_tables(prediction_rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, Optional[str]]:
    """Guarda predicciones largas y tabla pivote de votos por sujeto."""
    if not prediction_rows:
        return {"predictions": None, "votes": None}

    pred_df = pd.DataFrame(prediction_rows)
    pred_path = output_dir / "predicciones_por_sujeto.csv"
    pred_df.to_csv(pred_path, index=False)

    pred_df["vote_column"] = pred_df["method"] + "_" + pred_df["classifier"]
    votes = pred_df.pivot_table(
        index=["subject_id", "file_id", "true_diagnosis"],
        columns="vote_column",
        values="pred_diagnosis",
        aggfunc="first",
    ).reset_index()
    probs = pred_df.pivot_table(
        index=["subject_id", "file_id", "true_diagnosis"],
        columns="vote_column",
        values="prob_asd",
        aggfunc="first",
    ).reset_index()
    probs = probs.rename(
        columns={col: f"prob_asd_{col}" for col in probs.columns if col not in {"subject_id", "file_id", "true_diagnosis"}}
    )
    votes = votes.merge(probs, on=["subject_id", "file_id", "true_diagnosis"], how="left")
    vote_path = output_dir / "votos_por_sujeto.csv"
    votes.to_csv(vote_path, index=False)
    return {"predictions": str(pred_path), "votes": str(vote_path)}


def save_fold_metrics_table(fold_rows: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    """Guarda metricas por fold con medias y desviaciones estandar ya trazables."""
    if not fold_rows:
        return None
    path = output_dir / "metricas_por_fold.csv"
    pd.DataFrame(fold_rows).to_csv(path, index=False)
    return path


def mostrar_resultado_sujeto(output_dir: Path, file_id: str) -> pd.DataFrame:
    """
    Visor simple para notebook: muestra predicciones y matrices guardadas de un sujeto.

    Ejemplo:
        from script.pipeline_abide import mostrar_resultado_sujeto
        mostrar_resultado_sujeto(Path("resultados/pipeline_abide"), "Pitt_0050003")
    """
    output_dir = Path(output_dir)
    pred_path = output_dir / "predicciones_por_sujeto.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"No existe {pred_path}. Ejecuta run_pipeline primero.")

    pred_df = pd.read_csv(pred_path)
    subject_pred = pred_df[pred_df["file_id"] == file_id].copy()
    if subject_pred.empty:
        raise ValueError(f"No se encontraron predicciones para {file_id}.")

    matrix_files = sorted((output_dir / "matrices").glob(f"*/*{file_id}*.npy"))
    n = len(matrix_files)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
        for ax, path in zip(axes.ravel(), matrix_files):
            matrix = np.load(path)
            method = path.stem.replace(f"{file_id}_", "")
            vmax = np.nanpercentile(np.abs(matrix), 98)
            vmax = float(vmax) if vmax > 0 else 1.0
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(method)
            ax.set_xlabel("ROI")
            ax.set_ylabel("ROI")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Conectividad por metodo - {file_id}", y=1.02)
        fig.tight_layout()
        plt.show()

    return subject_pred[
        [
            "file_id",
            "true_diagnosis",
            "method",
            "connectivity_family",
            "classifier",
            "pred_diagnosis",
            "prob_asd",
            "correct",
        ]
    ].sort_values(["method", "classifier"])


def preparar_visor_atlas_sujeto(output_dir: Path, file_id: str) -> Dict[str, object]:
    """
    Prepara objetos para usar tus visores originales con atlas.

    Uso en notebook:
        from script.pipeline_abide import preparar_visor_atlas_sujeto
        from script.visor import visor_parcelacion

        v = preparar_visor_atlas_sujeto(Path("resultados/pipeline_atlas"), "Pitt_0050003")
        visor_parcelacion(
            v["fmri_data"], v["atlas_data"], v["roi_signal_cache"],
            v["roi_name_map"], v["roi_ids_all"], v["roi_sizes_all"]
        )
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Este visor requiere nibabel. Instala con: pip install nibabel") from exc

    output_dir = Path(output_dir)
    subject_dir = output_dir / "parcelacion_atlas" / file_id
    metadata_path = subject_dir / "roi_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe {metadata_path}. Ejecuta run_pipeline con source='nifti_atlas' primero."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fmri_path = Path(metadata["fmri_path"])
    atlas_data = np.load(metadata["atlas_data_path"])
    roi_signals_orig = np.load(subject_dir / "roi_signals_orig.npy")
    roi_signals_filt = np.load(subject_dir / "roi_signals_filt.npy")

    selected_rois = [int(r) for r in metadata["selected_rois"]]
    roi_names = [str(n) for n in metadata["roi_names"]]
    roi_sizes = {int(k): int(v) for k, v in metadata["roi_sizes"].items()}

    roi_signal_cache = {
        rid: (roi_signals_orig[:, idx], roi_signals_filt[:, idx])
        for idx, rid in enumerate(selected_rois)
    }
    roi_name_map = {rid: roi_names[idx] for idx, rid in enumerate(selected_rois)}

    fmri_img = nib.load(str(fmri_path))
    fmri_data = fmri_img.get_fdata(dtype=np.float32)

    return {
        "fmri_data": fmri_data,
        "atlas_data": atlas_data,
        "roi_signal_cache": roi_signal_cache,
        "roi_name_map": roi_name_map,
        "roi_ids_all": selected_rois,
        "roi_sizes_all": roi_sizes,
        "metadata": metadata,
    }


def _cargar_matriz_sujeto(output_dir: Path, file_id: str, method: str) -> np.ndarray:
    """Carga la matriz de conectividad guardada para un sujeto/metodo."""
    matches = sorted((Path(output_dir) / "matrices").glob(f"{method}_*rois/{file_id}_{method}.npy"))
    if not matches:
        raise FileNotFoundError(
            f"No encontre matriz para {file_id} metodo={method} en {Path(output_dir) / 'matrices'}"
        )
    return np.load(matches[-1])


def _roi_centroides_mni(output_dir: Path, file_id: str) -> Tuple[np.ndarray, List[str]]:
    """Calcula centroides de ROI en coordenadas del NIfTI para graficar conectomas."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Esta visualizacion requiere nibabel.") from exc

    output_dir = Path(output_dir)
    subject_dir = output_dir / "parcelacion_atlas" / file_id
    metadata_path = subject_dir / "roi_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe {metadata_path}. Esta figura requiere source='nifti_atlas'."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    atlas_data = np.load(metadata["atlas_data_path"])
    fmri_img = nib.load(metadata["fmri_path"])
    affine = fmri_img.affine

    selected_rois = [int(r) for r in metadata["selected_rois"]]
    roi_names = [str(n) for n in metadata["roi_names"]]
    coords = []

    for rid in selected_rois:
        vox = np.argwhere(atlas_data == rid)
        if len(vox) == 0:
            coords.append([np.nan, np.nan, np.nan])
            continue
        centroid_vox = vox.mean(axis=0)
        centroid_world = nib.affines.apply_affine(affine, centroid_vox)
        coords.append(centroid_world)

    coords = np.asarray(coords, dtype=float)
    valid = ~np.isnan(coords).any(axis=1)
    return coords[valid], [name for name, keep in zip(roi_names, valid) if keep]


def plot_conexiones_cerebro_sujeto(
    output_dir: Path,
    file_id: str,
    method: str = "pearson",
    top_n: int = 20,
    modo: str = "glass",
    guardar: bool = True,
) -> Optional[Path]:
    """
    Grafica conexiones ROI-ROI sobre el cerebro.

    Parametros
    ----------
    output_dir : Path
        Carpeta de resultados generada con source='nifti_atlas'.
    file_id : str
        Ejemplo: "Pitt_0050003".
    method : str
        pearson, partial, graphical_lasso, granger o pcmci.
    top_n : int
        Numero de conexiones mas fuertes a mostrar.
    modo : str
        "glass" usa nilearn.plotting.plot_connectome.
        "flechas3d" usa matplotlib 3D con flechas. Recomendado para Granger/PCMCI.
    guardar : bool
        Si True, guarda PNG en visor_conectividad/<file_id>/.

    Retorna
    -------
    Path o None
    """
    matrix = _cargar_matriz_sujeto(output_dir, file_id, method)
    coords, roi_names = _roi_centroides_mni(output_dir, file_id)
    n = min(matrix.shape[0], len(coords))
    matrix = matrix[:n, :n]
    coords = coords[:n]

    symmetric = bool(METHODS[method]["symmetric"])
    if symmetric:
        candidates = [(i, j, matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    else:
        candidates = [(i, j, matrix[i, j]) for i in range(n) for j in range(n) if i != j]

    candidates = sorted(candidates, key=lambda item: abs(item[2]), reverse=True)[:top_n]
    adjacency = np.zeros((n, n), dtype=float)
    for i, j, value in candidates:
        adjacency[i, j] = value
        if symmetric:
            adjacency[j, i] = value

    out_dir = Path(output_dir) / "visor_conectividad" / file_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cerebro_{method}_{modo}_top{top_n}.png"

    if modo == "glass":
        try:
            from nilearn import plotting
        except ImportError as exc:
            raise ImportError("modo='glass' requiere nilearn.") from exc

        display = plotting.plot_connectome(
            adjacency,
            coords,
            edge_threshold=None,
            node_size=18,
            title=f"{file_id} - {METHODS[method]['label']} top {top_n}",
            colorbar=True,
        )
        if guardar:
            display.savefig(str(out_path), dpi=160)
        plotting.show()
        return out_path if guardar else None

    if modo == "flechas3d":
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=28, c="black", alpha=0.75)

        vals = np.array([abs(v) for _, _, v in candidates])
        max_val = float(vals.max()) if len(vals) and vals.max() > 0 else 1.0
        for i, j, value in candidates:
            start = coords[i]
            end = coords[j]
            delta = end - start
            color = "tomato" if value > 0 else "steelblue"
            ax.quiver(
                start[0],
                start[1],
                start[2],
                delta[0],
                delta[1],
                delta[2],
                length=1.0,
                normalize=False,
                arrow_length_ratio=0.12,
                linewidth=0.6 + 2.4 * abs(value) / max_val,
                color=color,
                alpha=0.75,
            )

        ax.set_title(f"{file_id} - {METHODS[method]['label']} top {top_n}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.tight_layout()
        if guardar:
            fig.savefig(out_path, dpi=160)
        plt.show()
        return out_path if guardar else None

    raise ValueError("modo debe ser 'glass' o 'flechas3d'.")


def imprimir_resumen_resultados(output_dir: Path, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Imprime y devuelve tablas clave del pipeline.

    Incluye:
    - tabla comparativa de metricas
    - predicciones por sujeto
    - top conexiones por metodo segun |valor|
    """
    output_dir = Path(output_dir)
    table_path = output_dir / "tabla_comparativa_resultados.csv"
    pred_path = output_dir / "predicciones_por_sujeto.csv"
    edge_dir = output_dir / "conectividad_tablas"

    result: Dict[str, pd.DataFrame] = {}

    if table_path.exists():
        metrics = pd.read_csv(table_path)
        cols = [
            "method",
            "classifier",
            "status",
            "n_subjects",
            "n_features",
            "accuracy",
            "accuracy_fold_std",
            "balanced_accuracy",
            "balanced_accuracy_fold_std",
            "auc",
            "auc_fold_std",
            "f1",
            "f1_fold_std",
            "precision",
            "recall",
            "specificity",
        ]
        cols = [c for c in cols if c in metrics.columns]
        print("\n=== Tabla comparativa de clasificacion ===")
        print(metrics[cols].to_string(index=False))
        result["metricas"] = metrics

    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        print("\n=== Primeras predicciones por sujeto ===")
        pred_cols = [
            "file_id",
            "true_diagnosis",
            "method",
            "connectivity_family",
            "classifier",
            "pred_diagnosis",
            "prob_asd",
            "correct",
        ]
        print(pred[pred_cols].head(20).to_string(index=False))
        result["predicciones"] = pred

    if edge_dir.exists():
        top_frames = []
        print("\n=== Top conexiones por metodo ===")
        for edge_path in sorted(edge_dir.glob("*_edges.csv")):
            edges = pd.read_csv(edge_path)
            top = edges.sort_values("valor_abs", ascending=False).head(top_n)
            print(f"\n--- {edge_path.stem.replace('_edges', '')} ---")
            print(
                top[
                    [
                        "file_id",
                        "diagnosis",
                        "method",
                        "connectivity_family",
                        "roi_origen",
                        "roi_destino",
                        "valor_conectividad",
                    ]
                ].to_string(index=False)
            )
            top_frames.append(top)
        if top_frames:
            result["top_conexiones"] = pd.concat(top_frames, ignore_index=True)

    return result


def save_subject_manifest(subjects: Sequence[Subject], output_dir: Path) -> Path:
    """Guarda los sujetos incluidos y sus etiquetas."""
    rows = [
        {
            "subject_id": s.subject_id,
            "file_id": s.file_id,
            "label": s.label,
            "diagnosis": diagnosis_name(s.label),
            "site": s.site,
            "n_timepoints": s.roi_signals_z.shape[0],
            "n_rois": s.roi_signals_z.shape[1],
            "status": s.status,
            "error_note": s.error_note,
            "source_path": str(s.source_path),
        }
        for s in subjects
    ]
    path = output_dir / "sujetos_incluidos.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def append_processing_summary_rows(
    rows: List[Dict[str, object]],
    subjects: Sequence[Subject],
    method: str,
    classifier: str,
    metrics: Dict[str, object],
    status: str = "ok",
    error_note: str = "",
) -> None:
    """Agrega filas sujeto-metodo-clasificador para el CSV final solicitado."""
    for subject in subjects:
        rows.append(
            {
                "subject_id": subject.subject_id,
                "diagnostico": diagnosis_name(subject.label),
                "sitio": subject.site,
                "metodo_conectividad": method,
                "clasificador": classifier,
                "accuracy": metrics.get("accuracy", np.nan),
                "auc": metrics.get("auc", np.nan),
                "f1": metrics.get("f1", np.nan),
                "precision": metrics.get("precision", np.nan),
                "recall": metrics.get("recall", np.nan),
                "estado_procesamiento": status,
                "nota_error": error_note,
            }
        )


def save_final_processing_csv(rows: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    """Guarda el CSV final de trazabilidad y metricas por sujeto."""
    if not rows:
        return None
    path = output_dir / "resumen_final_sujetos_metricas.csv"
    columns = [
        "subject_id",
        "diagnostico",
        "sitio",
        "metodo_conectividad",
        "clasificador",
        "accuracy",
        "auc",
        "f1",
        "precision",
        "recall",
        "estado_procesamiento",
        "nota_error",
    ]
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, index=False)
    return path


def run_pipeline(
    roi_dir: Optional[Path] = None,
    phenotypic_csv: Optional[Path] = None,
    output_dir: Path = Path("resultados/pipeline_abide"),
    methods: Sequence[str] = ("pearson", "graphical_lasso", "lingam"),
    classifiers: Sequence[str] = ("svm", "rf"),
    source: str = "nifti_atlas",
    fmri_dir: Optional[Path] = None,
    atlas_name: str = ATLAS_DEFAULT,
    n_splits: int = 5,
    max_subjects: Optional[int] = None,
    n_per_group: Optional[int] = N_PER_GROUP,
    all_balanced: bool = False,
    all_available: bool = False,
    max_rois: Optional[int] = None,
    min_timepoints: Optional[int] = None,
    min_voxels: int = 100,
    maxlag: int = 2,
    tau_max: int = 2,
    lingam_random_state: int = 42,
    granger_lag_strategy: str = "min_q",
    pcmci_value_mode: str = "signed_logp",
    apply_bandpass: bool = True,
    cv_strategy: str = "group_site",
    cache: bool = True,
    save_individual_plots: bool = True,
    max_individual_plots: Optional[int] = 20,
) -> pd.DataFrame:
    """Ejecuta el pipeline completo y devuelve tabla comparativa."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if phenotypic_csv is None:
        raise ValueError("phenotypic_csv es obligatorio.")
    atlas_name = enforce_supported_atlas(atlas_name)
    if all_available:
        n_per_group = None
        max_subjects = None
        all_balanced = False

    if source == "nifti_atlas":
        if fmri_dir is None:
            raise ValueError("fmri_dir es obligatorio cuando source='nifti_atlas'.")
        subjects = load_subjects_from_nifti_atlas(
            fmri_dir=fmri_dir,
            phenotypic_csv=phenotypic_csv,
            output_dir=output_dir,
            atlas_name=atlas_name,
            max_subjects=max_subjects,
            n_per_group=n_per_group,
            all_balanced=all_balanced,
            max_rois=max_rois,
            min_voxels=min_voxels,
            apply_bandpass=apply_bandpass,
        )
    else:
        raise ValueError("source debe ser 'nifti_atlas'.")

    if min_timepoints is not None:
        n_before = len(subjects)
        labels_before = np.array([subject.label for subject in subjects])
        subjects = filter_subjects_by_timepoints(subjects, min_timepoints=min_timepoints)
        labels_after = np.array([subject.label for subject in subjects])
        print(
            f"Filtro min_timepoints>={min_timepoints}: {len(subjects)}/{n_before} sujetos conservados "
            f"| excluidos={n_before - len(subjects)} "
            f"| ASD {int(np.sum(labels_before == 1))}->{int(np.sum(labels_after == 1))} "
            f"| Control {int(np.sum(labels_before == 0))}->{int(np.sum(labels_after == 0))}"
        )

    validate_subjects(subjects, atlas_name)
    save_subject_manifest(subjects, output_dir)
    min_timepoints_rule = min_timepoints if min_timepoints is not None else subjects[0].roi_signals_z.shape[1] + 1
    timepoint_site_path = save_timepoint_site_table(
        subjects,
        output_dir,
        min_timepoints=min_timepoints_rule,
    )
    stage_paths = save_stage_outputs(subjects, output_dir)

    all_results: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    fold_metric_rows: List[Dict[str, object]] = []
    matrix_summary_rows: List[Dict[str, object]] = []
    processing_summary_rows: List[Dict[str, object]] = []
    subject_plot_rows: List[Dict[str, object]] = []
    edge_table_paths: Dict[str, str] = {}

    for method in methods:
        try:
            X, y, matrices, matrix_paths = build_method_dataset(
                subjects,
                method=method,
                output_dir=output_dir,
                maxlag=maxlag,
                tau_max=tau_max,
                lingam_random_state=lingam_random_state,
                granger_lag_strategy=granger_lag_strategy,
                pcmci_value_mode=pcmci_value_mode,
                cache=cache,
            )
            matrix_plot = plot_mean_matrix(matrices, method, output_dir)
            matrix_summary_rows.extend(summarize_connectivity_matrices(subjects, method, matrices))
            edge_table_paths[method] = str(save_connectivity_edge_table(subjects, method, matrices, output_dir))
        except Exception as exc:
            note = str(exc)
            print(f"[AVISO] {method}: fallo en conectividad/vectorizacion: {note}")
            for classifier in classifiers:
                table_rows.append(
                    {
                        "method": method,
                        "classifier": classifier,
                        "status": "failed_connectivity",
                        "n_subjects": len(subjects),
                        "n_asd": int(sum(s.label == 1 for s in subjects)),
                        "n_control": int(sum(s.label == 0 for s in subjects)),
                        "n_features": np.nan,
                        "n_splits": np.nan,
                        "accuracy": np.nan,
                        "auc": np.nan,
                        "f1": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "matrix_plot": "",
                        "note": note,
                    }
                )
                append_processing_summary_rows(
                    processing_summary_rows,
                    subjects,
                    method,
                    classifier,
                    metrics={},
                    status="failed_connectivity",
                    error_note=note,
                )
            continue

        if save_individual_plots:
            n_plots = len(subjects) if max_individual_plots is None else min(len(subjects), max_individual_plots)
            for subject, matrix, matrix_path in zip(subjects[:n_plots], matrices[:n_plots], matrix_paths[:n_plots]):
                plot_path = plot_subject_matrix(matrix, subject, method, output_dir)
                subject_plot_rows.append(
                    {
                        "subject_id": subject.subject_id,
                        "file_id": subject.file_id,
                        "method": method,
                        "matrix_npy": str(matrix_path),
                        "matrix_png": str(plot_path),
                    }
                )

        for classifier in classifiers:
            try:
                res = evaluate_classifier(
                    X,
                    y,
                    classifier,
                    n_splits=n_splits,
                    cv_strategy=cv_strategy,
                    groups=np.array([s.site for s in subjects]),
                )
                res["method"] = method
                all_results.append(res)
                append_subject_predictions(prediction_rows, subjects, method, classifier, res)
                append_fold_metric_rows(fold_metric_rows, method, classifier, res)
                append_processing_summary_rows(
                    processing_summary_rows,
                    subjects,
                    method,
                    classifier,
                    metrics=res,
                    status="ok",
                    error_note="",
                )

                table_rows.append(
                    {
                        "method": method,
                        "classifier": classifier,
                        "status": "ok",
                        "n_subjects": len(y),
                        "n_asd": int(np.sum(y == 1)),
                        "n_control": int(np.sum(y == 0)),
                        "n_features": X.shape[1],
                        "n_splits": res["n_splits"],
                        "accuracy": res["accuracy"],
                        "accuracy_fold_mean": res["accuracy_mean"],
                        "accuracy_fold_std": res["accuracy_std"],
                        "balanced_accuracy": res["balanced_accuracy"],
                        "balanced_accuracy_fold_mean": res["balanced_accuracy_mean"],
                        "balanced_accuracy_fold_std": res["balanced_accuracy_std"],
                        "auc": res["auc"],
                        "auc_fold_mean": res["auc_mean"],
                        "auc_fold_std": res["auc_std"],
                        "f1": res["f1"],
                        "f1_fold_mean": res["f1_mean"],
                        "f1_fold_std": res["f1_std"],
                        "precision": res["precision"],
                        "precision_fold_mean": res["precision_mean"],
                        "precision_fold_std": res["precision_std"],
                        "recall": res["recall"],
                        "recall_fold_mean": res["recall_mean"],
                        "recall_fold_std": res["recall_std"],
                        "specificity": res["specificity"],
                        "specificity_fold_mean": res["specificity_mean"],
                        "specificity_fold_std": res["specificity_std"],
                        "matrix_plot": str(matrix_plot),
                        "note": "",
                    }
                )
            except Exception as exc:
                note = str(exc)
                print(f"[AVISO] {method} + {classifier}: {note}")
                status = "skipped_insufficient_classes"
                table_rows.append(
                    {
                        "method": method,
                        "classifier": classifier,
                        "status": status,
                        "n_subjects": len(y),
                        "n_asd": int(np.sum(y == 1)),
                        "n_control": int(np.sum(y == 0)),
                        "n_features": X.shape[1],
                        "n_splits": np.nan,
                        "accuracy": np.nan,
                        "auc": np.nan,
                        "f1": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "matrix_plot": str(matrix_plot),
                        "note": note,
                    }
                )
                append_processing_summary_rows(
                    processing_summary_rows,
                    subjects,
                    method,
                    classifier,
                    metrics={},
                    status=status,
                    error_note=note,
                )

    roc_path = None
    if all_results:
        roc_path = plot_roc_curves(all_results, output_dir)
    prediction_paths = save_prediction_tables(prediction_rows, output_dir)
    fold_metrics_path = save_fold_metrics_table(fold_metric_rows, output_dir)
    matrix_summary_path = save_matrix_summary_table(matrix_summary_rows, output_dir)
    final_processing_path = save_final_processing_csv(processing_summary_rows, output_dir)
    subject_plot_path = None
    if subject_plot_rows:
        subject_plot_path = output_dir / "visor_conectividad" / "indice_visores.csv"
        pd.DataFrame(subject_plot_rows).to_csv(subject_plot_path, index=False)

    table = pd.DataFrame(table_rows)
    if {"auc", "accuracy"}.issubset(table.columns):
        table = table.sort_values(["auc", "accuracy"], ascending=False, na_position="last")
    table_path = output_dir / "tabla_comparativa_resultados.csv"
    table.to_csv(table_path, index=False)

    metadata = {
        "source": source,
        "roi_dir": str(roi_dir),
        "fmri_dir": str(fmri_dir),
        "phenotypic_csv": str(phenotypic_csv),
        "atlas_name": atlas_name,
        "methods": list(methods),
        "classifiers": list(classifiers),
        "max_subjects": max_subjects,
        "n_per_group": n_per_group,
        "all_balanced": all_balanced,
        "all_available": all_available,
        "max_rois": max_rois,
        "min_timepoints": min_timepoints,
        "maxlag": maxlag,
        "tau_max": tau_max,
        "lingam_random_state": lingam_random_state,
        "granger_lag_strategy": granger_lag_strategy,
        "pcmci_value_mode": pcmci_value_mode,
        "apply_bandpass": apply_bandpass,
        "cv_strategy": cv_strategy,
        "stage_outputs": stage_paths,
        "timepoint_site_summary_csv": str(timepoint_site_path) if timepoint_site_path is not None else None,
        "prediction_tables": prediction_paths,
        "fold_metrics_csv": str(fold_metrics_path) if fold_metrics_path is not None else None,
        "matrix_summary_csv": str(matrix_summary_path) if matrix_summary_path is not None else None,
        "final_processing_csv": str(final_processing_path) if final_processing_path is not None else None,
        "connectivity_edge_tables": edge_table_paths,
        "subject_plot_index": str(subject_plot_path) if subject_plot_path is not None else None,
        "save_individual_plots": save_individual_plots,
        "max_individual_plots": max_individual_plots,
        "roc_plot": str(roc_path) if roc_path is not None else None,
        "table": str(table_path),
    }
    (output_dir / "configuracion_pipeline.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nTabla comparativa:")
    print(table.to_string(index=False))
    if edge_table_paths:
        print("\nTablas de conectividad por metodo:")
        for method, path in edge_table_paths.items():
            print(f"  {method}: {path}")
    print(f"\nResultados guardados en: {output_dir}")
    return table


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline ABIDE fMRI ASD vs Control")
    parser.add_argument("--source", default="nifti_atlas", choices=["nifti_atlas"])
    parser.add_argument("--roi-dir", type=Path, default=None, help="No usado; se conserva por compatibilidad.")
    parser.add_argument("--fmri-dir", type=Path, default=None, help="Directorio con *_func_preproc.nii.gz")
    parser.add_argument("--phenotypic", type=Path, required=True, help="CSV fenotipico ABIDE")
    parser.add_argument("--output-dir", type=Path, default=Path("resultados/pipeline_schaefer_100"))
    parser.add_argument("--atlas-name", default=ATLAS_DEFAULT, choices=list(ATLAS_DISPONIBLES))
    parser.add_argument("--methods", nargs="+", default=["pearson", "graphical_lasso", "lingam"], choices=list(METHODS))
    parser.add_argument("--classifiers", nargs="+", default=["svm", "rf"], choices=list(CLASSIFIERS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-per-group", type=int, default=N_PER_GROUP, help="Sujetos por clase. Ej: 150 => 300 total.")
    parser.add_argument("--all-balanced", action="store_true", help="Usar todos los sujetos disponibles manteniendo balance ASD/Control.")
    parser.add_argument("--all-available", action="store_true", help="Usar todos los sujetos validos disponibles sin forzar balance de clases.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Compatibilidad: total balanceado; se ignora si --n-per-group esta definido.")
    parser.add_argument("--max-rois", type=int, default=None)
    parser.add_argument("--min-timepoints", type=int, default=None, help="Excluir sujetos con menos puntos temporales que este umbral.")
    parser.add_argument("--min-voxels", type=int, default=100)
    parser.add_argument("--maxlag", type=int, default=2, help="Lag maximo para Granger")
    parser.add_argument("--tau-max", type=int, default=2, help="Lag maximo para PCMCI")
    parser.add_argument("--lingam-random-state", type=int, default=42)
    parser.add_argument("--granger-lag-strategy", default="min_q", choices=["min_q", "min_p", "maxlag", "mean_p"])
    parser.add_argument("--pcmci-value-mode", default="signed_logp", choices=["signed_logp", "effect"])
    parser.add_argument("--skip-bandpass", action="store_true", help="No aplicar bandpass local; usar si los NIfTI estan en filt_*.")
    parser.add_argument("--cv-strategy", default="group_site", choices=["group_site", "stratified"])
    parser.add_argument("--no-cache", action="store_true", help="Recalcular matrices aunque existan")
    parser.add_argument("--no-individual-plots", action="store_true", help="No guardar heatmaps individuales tipo visor")
    parser.add_argument(
        "--max-individual-plots",
        type=int,
        default=20,
        help="Cantidad maxima de sujetos con PNG individual por metodo. Use -1 para todos.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(
        source=args.source,
        roi_dir=args.roi_dir,
        fmri_dir=args.fmri_dir,
        phenotypic_csv=args.phenotypic,
        output_dir=args.output_dir,
        atlas_name=args.atlas_name,
        methods=args.methods,
        classifiers=args.classifiers,
        n_splits=args.n_splits,
        max_subjects=args.max_subjects,
        n_per_group=args.n_per_group,
        all_balanced=args.all_balanced,
        all_available=args.all_available,
        max_rois=args.max_rois,
        min_timepoints=args.min_timepoints,
        min_voxels=args.min_voxels,
        maxlag=args.maxlag,
        tau_max=args.tau_max,
        lingam_random_state=args.lingam_random_state,
        granger_lag_strategy=args.granger_lag_strategy,
        pcmci_value_mode=args.pcmci_value_mode,
        apply_bandpass=not args.skip_bandpass,
        cv_strategy=args.cv_strategy,
        cache=not args.no_cache,
        save_individual_plots=not args.no_individual_plots,
        max_individual_plots=None if args.max_individual_plots == -1 else args.max_individual_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
