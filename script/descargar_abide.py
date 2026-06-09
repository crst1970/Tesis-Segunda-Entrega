"""
descargar_abide.py
------------------
Descarga sujetos ABIDE PCP en NIfTI preprocesado para usar con pipeline_abide.py.

Por defecto descarga 100 sujetos balanceados:
- 50 ASD
- 50 Control

Usa Nilearn:
    datasets.fetch_abide_pcp(
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["func_preproc"],
    )

Salida esperada:
    data/ABIDE_pcp/cpac/filt_noglobal/*_func_preproc.nii.gz

Ejemplo desde Pipeline_manual/notebooks:
    python script/descargar_abide.py --n-per-group 50 --derivative func_preproc
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import requests
from nilearn import datasets

N_PER_GROUP = 100


def download_group(
    data_dir: Path,
    dx_group: int,
    n_subjects: Optional[int],
    pipeline: str,
    derivative: str,
    band_pass_filtering: bool,
    global_signal_regression: bool,
    retries: int = 5,
    retry_delay: float = 30.0,
):
    """Descarga un grupo diagnostico de ABIDE PCP."""
    label = "ASD" if dx_group == 1 else "Control"
    amount = "todos los disponibles" if n_subjects is None else str(n_subjects)
    print(f"\nDescargando {amount} sujetos {label}...")

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return datasets.fetch_abide_pcp(
                data_dir=data_dir,
                n_subjects=n_subjects,
                pipeline=pipeline,
                band_pass_filtering=band_pass_filtering,
                global_signal_regression=global_signal_regression,
                derivatives=[derivative],
                quality_checked=True,
                DX_GROUP=dx_group,
                verbose=1,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(
                f"[AVISO] Timeout/conexion descargando {label} "
                f"(intento {attempt}/{retries}). Reintentando en {retry_delay:.0f}s..."
            )
            time.sleep(retry_delay)

    raise RuntimeError(
        f"No se pudo descargar el grupo {label} despues de {retries} intentos. "
        "Puedes volver a ejecutar la celda; Nilearn reusa lo ya descargado."
    ) from last_error


def copy_phenotypic_if_needed(data_dir: Path, repo_phenotypic: Optional[Path]) -> Optional[Path]:
    """Copia el CSV fenotipico descargado al path usado por los notebooks, si falta."""
    downloaded = data_dir / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"
    if not downloaded.exists() or repo_phenotypic is None:
        return downloaded if downloaded.exists() else None

    repo_phenotypic.parent.mkdir(parents=True, exist_ok=True)
    if not repo_phenotypic.exists():
        shutil.copy2(downloaded, repo_phenotypic)
        print(f"Fenotipico copiado a: {repo_phenotypic}")
    return repo_phenotypic


def summarize_download(data_leaf_dir: Path, phenotypic_csv: Path, derivative: str) -> pd.DataFrame:
    """Resume sujetos descargados y balance de clases."""
    if derivative != "func_preproc":
        raise ValueError(
            "Este proyecto solo resume derivative='func_preproc'. "
            "Las senales ROI deben extraerse localmente con atlas Destrieux."
        )
    files = sorted(data_leaf_dir.glob("*_func_preproc.nii.gz"))
    if not files:
        files = sorted(data_leaf_dir.glob("*"))

    phen = pd.read_csv(phenotypic_csv)
    rows = []
    seen_file_ids = set()
    seen_subject_ids = set()
    for path in files:
        file_id = path.name.replace("_func_preproc.nii.gz", "")
        if file_id in seen_file_ids:
            rows.append({
                "file": path.name,
                "file_id": file_id,
                "subject_id": "",
                "site": "",
                "label": None,
                "diagnosis": "duplicado",
                "status": "skipped_duplicate_file_id",
                "error_note": "FILE_ID duplicado",
            })
            continue
        numeric = "".join(ch for ch in file_id if ch.isdigit())
        candidates = {file_id, numeric, str(int(numeric)) if numeric else ""}
        hit = phen[
            phen["FILE_ID"].astype(str).isin(candidates)
            | phen["SUB_ID"].astype(str).isin(candidates)
        ]
        if hit.empty:
            subject_id = ""
            site = ""
            label = None
            diagnosis = "sin_etiqueta"
            status = "skipped_missing_label"
            error_note = "No aparece en SUB_ID/FILE_ID del fenotipico"
        else:
            row = hit.iloc[0]
            subject_id = str(row["SUB_ID"])
            site = str(row["SITE_ID"]) if "SITE_ID" in hit.columns else ""
            if subject_id in seen_subject_ids:
                label = None
                diagnosis = "duplicado"
                status = "skipped_duplicate_subject_id"
                error_note = f"SUB_ID duplicado: {subject_id}"
            else:
                dx = int(row["DX_GROUP"])
                label = 1 if dx == 1 else 0
                diagnosis = "ASD" if label == 1 else "Control"
                status = "ok"
                error_note = ""
                seen_file_ids.add(file_id)
                seen_subject_ids.add(subject_id)
        rows.append({
            "file": path.name,
            "file_id": file_id,
            "subject_id": subject_id,
            "site": site,
            "label": label,
            "diagnosis": diagnosis,
            "status": status,
            "error_note": error_note,
        })

    summary = pd.DataFrame(rows)
    print("\nResumen de archivos ROI disponibles:")
    print(summary["diagnosis"].value_counts(dropna=False).to_string())
    print(f"Total archivos ROI: {len(summary)}")
    return summary


def run_download(
    data_dir: Path,
    n_asd: Optional[int] = N_PER_GROUP,
    n_control: Optional[int] = N_PER_GROUP,
    n_per_group: Optional[int] = None,
    all_balanced: bool = False,
    all_available: bool = False,
    pipeline: str = "cpac",
    derivative: str = "func_preproc",
    band_pass_filtering: bool = True,
    global_signal_regression: bool = False,
    repo_phenotypic: Optional[Path] = None,
    download_retries: int = 5,
    retry_delay: float = 30.0,
) -> pd.DataFrame:
    """Descarga ABIDE balanceado y devuelve resumen de archivos."""
    if derivative != "func_preproc":
        raise ValueError(
            "Este proyecto usa solo NIfTI func_preproc + atlas Destrieux. "
            "No se permiten derivados ROI precomputados."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    if all_available:
        n_asd = None
        n_control = None
    elif all_balanced:
        n_asd = None
        n_control = None
    elif n_per_group is not None:
        n_asd = n_per_group
        n_control = n_per_group

    if n_asd is None or n_asd > 0:
        download_group(
            data_dir=data_dir,
            dx_group=1,
            n_subjects=n_asd,
            pipeline=pipeline,
            derivative=derivative,
            band_pass_filtering=band_pass_filtering,
            global_signal_regression=global_signal_regression,
            retries=download_retries,
            retry_delay=retry_delay,
        )
    if n_control is None or n_control > 0:
        download_group(
            data_dir=data_dir,
            dx_group=2,
            n_subjects=n_control,
            pipeline=pipeline,
            derivative=derivative,
            band_pass_filtering=band_pass_filtering,
            global_signal_regression=global_signal_regression,
            retries=download_retries,
            retry_delay=retry_delay,
        )

    filt = "filt" if band_pass_filtering else "nofilt"
    gsr = "global" if global_signal_regression else "noglobal"
    data_leaf_dir = data_dir / "ABIDE_pcp" / pipeline / f"{filt}_{gsr}"

    phenotypic_csv = copy_phenotypic_if_needed(data_dir, repo_phenotypic)
    if phenotypic_csv is None:
        raise FileNotFoundError("No se encontro Phenotypic_V1_0b_preprocessed1.csv despues de la descarga.")

    summary = summarize_download(data_leaf_dir, phenotypic_csv, derivative)
    summary_path = data_leaf_dir / f"resumen_descarga_{derivative}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Resumen guardado en: {summary_path}")
    if derivative == "func_preproc":
        print(f"Usa este fmri_dir en el pipeline atlas: {data_leaf_dir}")
    else:
        print(f"Usa este roi_dir en el pipeline: {data_leaf_dir}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga ABIDE PCP balanceado para pruebas ASD vs Control")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directorio de datos Nilearn/ABIDE")
    parser.add_argument("--n-per-group", type=int, default=N_PER_GROUP, help="Sujetos por clase. Ej: 150 => 300 total.")
    parser.add_argument("--all-balanced", action="store_true", help="Descargar todos los disponibles por clase.")
    parser.add_argument("--all-available", action="store_true", help="Descargar todos los disponibles por clase.")
    parser.add_argument("--n-asd", type=int, default=None, help="Compatibilidad: sujetos ASD; se ignora si --n-per-group esta definido.")
    parser.add_argument("--n-control", type=int, default=None, help="Compatibilidad: sujetos Control; se ignora si --n-per-group esta definido.")
    parser.add_argument("--pipeline", default="cpac", choices=["cpac", "ccs", "dparsf", "niak"])
    parser.add_argument("--derivative", default="func_preproc", choices=["func_preproc"])
    parser.add_argument("--no-bandpass", action="store_true")
    parser.add_argument("--global-signal-regression", action="store_true")
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=30.0, help="Segundos entre reintentos de descarga")
    parser.add_argument(
        "--repo-phenotypic",
        type=Path,
        default=Path("data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_download(
        data_dir=args.data_dir,
        n_asd=args.n_asd,
        n_control=args.n_control,
        n_per_group=args.n_per_group,
        all_balanced=args.all_balanced,
        all_available=args.all_available,
        pipeline=args.pipeline,
        derivative=args.derivative,
        band_pass_filtering=not args.no_bandpass,
        global_signal_regression=args.global_signal_regression,
        repo_phenotypic=args.repo_phenotypic,
        download_retries=args.download_retries,
        retry_delay=args.retry_delay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
