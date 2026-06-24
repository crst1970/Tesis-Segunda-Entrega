from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ALIASES: Mapping[str, str] = {
    "correlacion": "pearson",
    "lasso": "graphical_lasso",
    "lingam": "lingam",
}

PUBLIC_LABELS: Mapping[str, str] = {
    "correlacion": "Correlación de Pearson",
    "lasso": "Graphical Lasso",
    "lingam": "LiNGAM",
}

DIRECTED_METHODS = {"lingam"}
SPARSITY_SUPPORT_THRESHOLD = 1e-3

CONNECTOME_PRESENTATION_TITLES: Mapping[str, str] = {
    "correlacion": "Conectomas promedio y diferencias entre grupos — Correlación de Pearson",
    "lasso": "Conectomas promedio y diferencias entre grupos — Graphical Lasso",
    "lingam": "Conectomas promedio y diferencias entre grupos — LiNGAM",
}

SPARSITY_SUPPORT_LABELS: Mapping[str, str] = {
    "correlacion": "Correlación de Pearson",
    "lasso": "Graphical Lasso",
    "lingam": "LiNGAM",
}

SPARSITY_SUPPORT_NOTES: Mapping[str, str] = {
    "correlacion": "Red densa; esperable en correlación funcional.",
    "lasso": "Red más dispersa; refleja regularización y asociaciones directas.",
    "lingam": "Red dirigida exploratoria; esparsidad dependiente del umbral.",
}

NETWORK_ORDER = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
    "unknown",
]

NETWORK_PUBLIC_LABELS: Mapping[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotora",
    "DorsAttn": "Atención dorsal",
    "SalVentAttn": "Saliencia/atención ventral",
    "Limbic": "Límbica",
    "Cont": "Control/frontoparietal",
    "Default": "Modo por defecto",
    "unknown": "No identificada",
}

NETWORK_LITERATURE_CONTEXT: Mapping[str, Mapping[str, object]] = {
    "Default": {
        "match_level": "high",
        "note": (
            "Red default/DMN, incluyendo precuneus-PCC y PFC medial/lateral, "
            "es un foco recurrente en conectividad funcional TEA."
        ),
        "citation_keys": ("Assaf2010_DMN", "DiMartino2014_ABIDE"),
    },
    "SalVentAttn": {
        "match_level": "high",
        "note": (
            "Red de saliencia/atencion ventral; ABIDE reporta loci comunes "
            "en insula media/posterior, cercanos a este sistema."
        ),
        "citation_keys": ("DiMartino2014_ABIDE",),
    },
    "Cont": {
        "match_level": "moderate",
        "note": (
            "Red de control/frontoparietal. Es compatible con literatura de "
            "alteraciones corticocorticales distribuidas, aunque no es una "
            "coincidencia anatomica especifica por si sola."
        ),
        "citation_keys": ("DiMartino2014_ABIDE",),
    },
    "DorsAttn": {
        "match_level": "moderate",
        "note": (
            "Red de atencion dorsal. Se interpreta como parte de patrones "
            "task-positive/corticocorticales distribuidos, no como marcador "
            "especifico aislado."
        ),
        "citation_keys": ("DiMartino2014_ABIDE",),
    },
    "Limbic": {
        "match_level": "moderate",
        "note": (
            "Red limbica/temporal. Puede ser relevante para procesamiento "
            "socioemocional, pero requiere cautela por la resolucion del atlas."
        ),
        "citation_keys": ("DiMartino2014_ABIDE",),
    },
    "SomMot": {
        "match_level": "exploratory",
        "note": (
            "Red somatomotora. Resultado plausible en TEA, pero aqui se reporta "
            "como hallazgo descriptivo de la cohorte."
        ),
        "citation_keys": ("DiMartino2014_ABIDE",),
    },
    "Vis": {
        "match_level": "exploratory",
        "note": (
            "Red visual. Resultado plausible en TEA, pero aqui se reporta "
            "como hallazgo descriptivo de la cohorte."
        ),
        "citation_keys": ("DiMartino2014_ABIDE",),
    },
}

LITERATURE_REFERENCES = [
    {
        "citation_key": "DiMartino2014_ABIDE",
        "short_reference": "Di Martino et al., 2014, Molecular Psychiatry",
        "url": "https://www.nature.com/articles/mp201378",
        "relevance": (
            "ABIDE R-fMRI: reporta hipo e hiperconectividad en TEA, con "
            "hipoconectividad predominante y loci comunes en insula posterior/PCC."
        ),
    },
    {
        "citation_key": "Assaf2010_DMN",
        "short_reference": "Assaf et al., 2010, NeuroImage",
        "url": "https://pubmed.ncbi.nlm.nih.gov/20621638/",
        "relevance": (
            "DMN en TEA: conectividad reducida entre precuneus y mPFC/ACC, "
            "relacionada con severidad social/comunicativa."
        ),
    },
]


@dataclass
class MethodData:
    method: str
    source_method: str
    matrices: np.ndarray
    directed: bool
    roi_names: List[str]
    manifest: pd.DataFrame


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _public_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in METHOD_ALIASES:
        valid = ", ".join(METHOD_ALIASES)
        raise ValueError(f"Metodo v9 no soportado: {method!r}. Usa: {valid}")
    return normalized


def _source_method(method: str) -> str:
    return METHOD_ALIASES[_public_method(method)]


def load_subject_manifest(source_dir: str | Path) -> pd.DataFrame:
    source_dir = _as_path(source_dir)
    path = source_dir / "sujetos_incluidos.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No encontre {path}. Ejecuta primero pipeline_v8.ipynb para generar la corrida base."
        )

    manifest = pd.read_csv(path)
    required = {"file_id", "label", "diagnosis", "n_timepoints", "n_rois"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{path} no tiene columnas requeridas: {missing}")

    manifest = manifest.copy()
    manifest["file_id"] = manifest["file_id"].astype(str)
    manifest["label"] = manifest["label"].astype(int)
    if manifest["file_id"].duplicated().any():
        duplicates = manifest.loc[manifest["file_id"].duplicated(), "file_id"].tolist()
        raise ValueError(f"FILE_ID duplicados en la cohorte: {duplicates[:5]}")
    if not set(manifest["label"].unique()).issubset({0, 1}):
        raise ValueError("La cohorte contiene etiquetas distintas de 0=Control y 1=TEA.")
    manifest["group"] = np.where(manifest["label"] == 1, "TEA", "Control")

    n_rois = manifest["n_rois"].dropna().astype(int).unique()
    if len(n_rois) != 1:
        raise ValueError(f"La cohorte mezcla cantidades de ROIs: {sorted(n_rois)}")

    return manifest


def _read_roi_names(source_dir: Path, source_method: str, n_rois: int) -> List[str]:
    parcel_root = source_dir / "parcelacion_atlas"
    if parcel_root.exists():
        for metadata_path in sorted(parcel_root.glob("*/roi_metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metadata_names = metadata.get("roi_names")
            if isinstance(metadata_names, list) and len(metadata_names) >= n_rois:
                return [str(name) for name in metadata_names[:n_rois]]

    feature_path = source_dir / "feature_maps" / f"{source_method}_feature_map.csv"
    if not feature_path.exists():
        return [f"ROI_{i + 1:03d}" for i in range(n_rois)]

    fmap = pd.read_csv(feature_path)
    required = {"roi_origen_idx", "roi_origen", "roi_destino_idx", "roi_destino"}
    if not required.issubset(fmap.columns):
        return [f"ROI_{i + 1:03d}" for i in range(n_rois)]

    names = [f"ROI_{i + 1:03d}" for i in range(n_rois)]
    for idx_col, name_col in [("roi_origen_idx", "roi_origen"), ("roi_destino_idx", "roi_destino")]:
        first_by_idx = fmap.drop_duplicates(idx_col)
        for _, row in first_by_idx.iterrows():
            idx = int(row[idx_col])
            if 0 <= idx < n_rois:
                names[idx] = str(row[name_col])
    return names


def _find_parcellation_subject_dir(source_dir: Path, file_id: str | None = None) -> Path:
    parcel_root = source_dir / "parcelacion_atlas"
    if not parcel_root.exists():
        raise FileNotFoundError(f"No encontre carpeta de parcelacion: {parcel_root}")

    if file_id:
        subject_dir = parcel_root / str(file_id)
        if not subject_dir.exists():
            raise FileNotFoundError(f"No encontre parcelacion para file_id={file_id}: {subject_dir}")
        return subject_dir

    for subject_dir in sorted(path for path in parcel_root.iterdir() if path.is_dir()):
        required = [
            subject_dir / "atlas_resampled.npy",
            subject_dir / "roi_signals_filt.npy",
            subject_dir / "roi_metadata.json",
        ]
        if all(path.exists() for path in required):
            return subject_dir

    raise FileNotFoundError(f"No encontre sujetos parcelados completos en {parcel_root}")


def _select_roi_position(roi_names: Sequence[str], roi_name_query: str | None = None) -> int:
    if roi_name_query:
        query = str(roi_name_query)
        if query in roi_names:
            return list(roi_names).index(query)

        query_lower = query.lower()
        for idx, name in enumerate(roi_names):
            if query_lower in str(name).lower():
                return idx

    default_queries = ["Default_pCunPCC_2", "Default_pCunPCC_1", "Default"]
    for query in default_queries:
        query_lower = query.lower()
        for idx, name in enumerate(roi_names):
            if query_lower in str(name).lower():
                return idx

    return 0


def _subject_fmri_path(source_dir: Path, file_id: str) -> Path | None:
    trace_path = source_dir / "trazabilidad_parcelacion_filtrado.csv"
    if not trace_path.exists():
        return None

    try:
        trace = pd.read_csv(trace_path, usecols=["file_id", "fmri_path"])
    except Exception:
        return None

    matches = trace.loc[trace["file_id"].astype(str).eq(str(file_id)), "fmri_path"]
    if matches.empty:
        return None

    fmri_path = Path(str(matches.iloc[0]))
    return fmri_path if fmri_path.exists() else None


def _fmri_background_slice(
    source_dir: Path,
    file_id: str,
    z_index: int,
    atlas_shape: tuple[int, ...],
) -> tuple[np.ndarray, Path] | None:
    fmri_path = _subject_fmri_path(source_dir, file_id)
    if fmri_path is None:
        return None

    try:
        import nibabel as nib
    except Exception:
        return None

    try:
        image = nib.load(str(fmri_path))
        if tuple(image.shape[:3]) != tuple(atlas_shape[:3]):
            return None

        if len(image.shape) == 4:
            n_for_mean = min(30, int(image.shape[3]))
            data = np.asarray(image.dataobj[..., :n_for_mean], dtype=np.float32)
            volume = np.nanmean(data, axis=3)
        else:
            volume = np.asarray(image.dataobj, dtype=np.float32)
    except Exception:
        return None

    background = np.take(volume, z_index, axis=2).T
    finite = np.isfinite(background)
    nonzero = finite & (background != 0)
    values = background[nonzero] if np.any(nonzero) else background[finite]
    if values.size == 0:
        return None

    low, high = np.nanpercentile(values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if low == high:
        return None

    normalized = (background - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~nonzero] = np.nan
    return normalized, fmri_path


def create_roi_bold_extraction_example(
    source_dir: str | Path,
    output_dir: str | Path,
    file_id: str | None = None,
    roi_name_query: str | None = "7Networks_LH_Default_pCunPCC_2",
) -> Dict[str, object]:
    """Create a methodological figure for ROI parcellation and BOLD extraction.

    The function uses cached outputs from the base pipeline. It does not recompute
    connectivity matrices and does not add statistical results.
    """

    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    subject_dir = _find_parcellation_subject_dir(source_dir, file_id=file_id)
    metadata_path = subject_dir / "roi_metadata.json"
    atlas_path = subject_dir / "atlas_resampled.npy"
    signals_path = subject_dir / "roi_signals_filt.npy"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    roi_names = [str(name) for name in metadata.get("roi_names", [])]
    selected_rois = [int(roi) for roi in metadata.get("selected_rois", [])]
    if not roi_names:
        raise ValueError(f"{metadata_path} no contiene roi_names.")

    atlas = np.asarray(np.load(atlas_path))
    signals = np.asarray(np.load(signals_path), dtype=float)
    if signals.ndim != 2:
        raise ValueError(f"Senales ROI con shape inesperado en {signals_path}: {signals.shape}")
    if signals.shape[1] < len(roi_names):
        raise ValueError(
            f"{signals_path} tiene {signals.shape[1]} ROIs, pero la metadata lista {len(roi_names)}."
        )

    roi_pos = _select_roi_position(roi_names, roi_name_query=roi_name_query)
    roi_name = roi_names[roi_pos]
    roi_label = selected_rois[roi_pos] if roi_pos < len(selected_rois) else roi_pos + 1

    coords = np.argwhere(atlas == roi_label)
    if coords.size == 0 and roi_label != roi_pos + 1:
        roi_label = roi_pos + 1
        coords = np.argwhere(atlas == roi_label)
    if coords.size == 0:
        raise ValueError(f"No encontre voxeles para la ROI {roi_name} con etiqueta {roi_label}.")

    centroid = np.rint(coords.mean(axis=0)).astype(int)
    z_index = int(centroid[2])
    atlas_label_slice = np.take(atlas, z_index, axis=2).T
    brain_slice = atlas_label_slice > 0
    roi_slice = np.take(atlas == roi_label, z_index, axis=2).T
    original_signals_path = subject_dir / "roi_signals_orig.npy"
    original_signals = (
        np.asarray(np.load(original_signals_path), dtype=float)
        if original_signals_path.exists()
        else signals
    )
    if original_signals.shape != signals.shape:
        original_signals = signals

    original_signal = original_signals[:, roi_pos]
    pipeline_signal = signals[:, roi_pos]
    signals_are_equal = bool(np.allclose(original_signal, pipeline_signal, equal_nan=True))
    n_timepoints = int(signals.shape[0])
    n_rois = int(signals.shape[1])
    tr = float(metadata.get("tr", 1.0) or 1.0)
    volumes = np.arange(n_timepoints)
    subject_label = metadata.get("file_id", subject_dir.name)
    fmri_background = _fmri_background_slice(source_dir, str(subject_label), z_index, atlas.shape)
    zscore_signals_path = source_dir / "filtrado_zscore" / f"{subject_label}_roi_signals_z.npy"
    if zscore_signals_path.exists():
        zscore_signals = np.asarray(np.load(zscore_signals_path), dtype=float)
        if zscore_signals.shape != signals.shape:
            zscore_signals = signals
            zscore_signals_path = signals_path
    else:
        zscore_signals = signals
        zscore_signals_path = signals_path

    heatmap_limit = float(np.nanpercentile(np.abs(zscore_signals), 98))
    if not np.isfinite(heatmap_limit) or heatmap_limit <= 0:
        heatmap_limit = 1.0

    figure_path = figures_dir / "parcelacion_roi_bold_example.png"
    fig = plt.figure(
        figsize=(14.2, 4.35),
        dpi=170,
    )
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.92, 1.26, 1.36],
        wspace=0.24,
    )
    ax_atlas = fig.add_subplot(grid[0, 0])
    ax_signal = fig.add_subplot(grid[0, 1])
    ax_roi_time = fig.add_subplot(grid[0, 2])

    if fmri_background is None:
        background = np.full(atlas_label_slice.shape, np.nan, dtype=float)
        background[brain_slice] = 0.42 + 0.36 * ((atlas_label_slice[brain_slice] % 9) / 8.0)
        fmri_background_path = None
    else:
        background, fmri_background_path = fmri_background

    ax_atlas.set_facecolor("0.66")
    ax_atlas.imshow(
        background,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
    )

    parcellation_cmap = plt.get_cmap("tab20", 20)
    rgba = parcellation_cmap(((atlas_label_slice.astype(int) - 1) % 20).clip(min=0))
    rgba[..., 3] = np.where(brain_slice, 0.58, 0.0)
    rgba[roi_slice] = np.array([1.0, 0.78, 0.06, 0.96])
    ax_atlas.imshow(rgba, origin="lower", interpolation="nearest")
    ax_atlas.contour(roi_slice.astype(float), levels=[0.5], colors=["black"], linewidths=1.1)
    ax_atlas.contour(roi_slice.astype(float), levels=[0.5], colors=["white"], linewidths=0.55)
    ax_atlas.set_title("A. ROI en atlas", fontsize=9, loc="left")
    ax_atlas.set_xlabel("x", fontsize=8)
    ax_atlas.set_ylabel("y", fontsize=8)
    ax_atlas.tick_params(labelsize=7)
    ax_atlas.text(
        0.99,
        1.02,
        f"z={z_index} | ROI {roi_pos + 1}",
        transform=ax_atlas.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    ax_signal.plot(
        volumes,
        original_signal,
        color="#6aa0d8",
        linewidth=2.0,
        alpha=0.34,
        label="Original",
    )
    ax_signal.plot(
        volumes,
        pipeline_signal,
        color="#ff8a65",
        linewidth=1.2,
        alpha=0.95,
        label=(
            "Entrada conectividad (coincide)"
            if signals_are_equal
            else "Entrada conectividad"
        ),
    )
    ax_signal.set_title("B. Senal BOLD", fontsize=9, loc="left")
    ax_signal.set_xlabel("")
    ax_signal.set_ylabel("Amplitud BOLD", fontsize=8)
    ax_signal.tick_params(labelsize=7)
    ax_signal.grid(alpha=0.18)
    ax_signal.legend(loc="upper left", fontsize=7, frameon=True)
    ax_signal.text(
        0.99,
        1.02,
        f"ROI {roi_pos + 1} | T={n_timepoints}",
        transform=ax_signal.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    heatmap = ax_roi_time.imshow(
        zscore_signals.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-heatmap_limit,
        vmax=heatmap_limit,
        origin="lower",
        interpolation="nearest",
    )
    ax_roi_time.axhline(roi_pos, color="#ffd166", linewidth=1.3)
    ax_roi_time.set_title("C. ROIs x tiempo", fontsize=9, loc="left")
    ax_roi_time.set_xlabel("Tiempo (volumenes)", fontsize=8)
    ax_roi_time.set_ylabel("ROI Schaefer", fontsize=8)
    xticks = np.linspace(0, n_timepoints - 1, 5, dtype=int)
    ax_roi_time.set_xticks(xticks)
    ax_roi_time.set_xticklabels([str(idx) for idx in xticks])
    y_ticks = sorted({0, roi_pos, n_rois - 1})
    ax_roi_time.set_yticks(y_ticks)
    ax_roi_time.set_yticklabels([str(idx + 1) for idx in y_ticks])
    ax_roi_time.tick_params(labelsize=7)
    colorbar = fig.colorbar(heatmap, ax=ax_roi_time, fraction=0.018, pad=0.012)
    colorbar.set_label("z-score BOLD", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)

    atlas_name = metadata.get("atlas_name", "atlas no especificado")
    fig.suptitle("Extraccion de senales por ROI", fontsize=11, y=1.03)
    fig.text(
        0.5,
        0.965,
        f"{subject_label} | Schaefer 100 | ROI {roi_pos + 1} | T={n_timepoints}",
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    example_metadata = pd.DataFrame(
        [
            {
                "file_id": subject_label,
                "subject_id": metadata.get("subject_id"),
                "diagnosis": metadata.get("diagnosis"),
                "site": metadata.get("site"),
                "atlas_name": atlas_name,
                "roi_index_1based": roi_pos + 1,
                "roi_label_in_atlas": roi_label,
                "roi_name": roi_name,
                "tr": tr,
                "n_timepoints": n_timepoints,
                "n_rois": n_rois,
                "signal_source": str(signals_path),
                "original_signal_source": str(original_signals_path),
                "roi_time_matrix_source": str(zscore_signals_path),
                "signals_are_equal": signals_are_equal,
                "signal_processing_note": "La corrida principal no aplica bandpass local; el panel ROIs x tiempo usa la matriz z-score final para conectividad.",
                "atlas_source": str(atlas_path),
                "fmri_background_source": str(fmri_background_path) if fmri_background_path else "",
                "figure_path": str(figure_path),
            }
        ]
    )
    example_metadata_path = tables_dir / "parcelacion_roi_bold_example_metadata.csv"
    example_metadata.to_csv(example_metadata_path, index=False)

    return {
        "figure_path": figure_path,
        "metadata_path": example_metadata_path,
        "file_id": subject_label,
        "roi_index_1based": roi_pos + 1,
        "roi_label_in_atlas": roi_label,
        "roi_name": roi_name,
        "n_timepoints": n_timepoints,
        "n_rois": n_rois,
        "signals_are_equal": signals_are_equal,
    }


def _roi_hemisphere(roi_name: str) -> str:
    parts = str(roi_name).split("_")
    if len(parts) >= 3 and parts[0] == "7Networks" and parts[1] in {"LH", "RH"}:
        return parts[1]
    return "unknown"


def _roi_network(roi_name: str) -> str:
    parts = str(roi_name).split("_")
    if len(parts) >= 3 and parts[0] == "7Networks" and parts[1] in {"LH", "RH"}:
        return parts[2]
    return "unknown"


def _roi_region(roi_name: str) -> str:
    parts = str(roi_name).split("_")
    if len(parts) >= 4 and parts[0] == "7Networks" and parts[1] in {"LH", "RH"}:
        return "_".join(parts[3:])
    return str(roi_name)


def _network_pair(network_i: str, network_j: str) -> str:
    order = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    a, b = sorted([network_i, network_j], key=lambda n: order.get(n, len(order)))
    return f"{a}--{b}"


def _literature_context_for_pair(network_i: str, network_j: str) -> Dict[str, str]:
    contexts = []
    for network in dict.fromkeys([network_i, network_j]):
        context = NETWORK_LITERATURE_CONTEXT.get(network)
        if context is not None:
            contexts.append(context)

    if not contexts:
        return {
            "literature_match_level": "unknown",
            "literature_note": "No hay mapeo de red confiable para esta ROI.",
            "citation_keys": "",
        }

    levels = [str(context["match_level"]) for context in contexts]
    if "high" in levels:
        match_level = "high"
    elif "moderate" in levels:
        match_level = "moderate"
    else:
        match_level = "exploratory"

    notes = [str(context["note"]) for context in contexts]
    citation_keys = sorted(
        {
            str(key)
            for context in contexts
            for key in context.get("citation_keys", ())
        }
    )
    return {
        "literature_match_level": match_level,
        "literature_note": " ".join(dict.fromkeys(notes)),
        "citation_keys": ";".join(citation_keys),
    }


def _matrix_path(source_dir: Path, source_method: str, n_rois: int, file_id: str) -> Path:
    return source_dir / "matrices" / f"{source_method}_{n_rois}rois" / f"{file_id}_{source_method}.npy"


def load_method_data(source_dir: str | Path, method: str, manifest: pd.DataFrame | None = None) -> MethodData:
    source_dir = _as_path(source_dir)
    public_method = _public_method(method)
    source_method = _source_method(public_method)
    if manifest is None:
        manifest = load_subject_manifest(source_dir)
    else:
        manifest = manifest.copy()

    n_rois = int(manifest["n_rois"].iloc[0])
    matrix_dir = source_dir / "matrices" / f"{source_method}_{n_rois}rois"
    if not matrix_dir.exists():
        raise FileNotFoundError(
            f"No encontre matrices para {public_method}: {matrix_dir}. Ejecuta v8 con ese metodo primero."
        )

    matrices = []
    missing = []
    for file_id in manifest["file_id"].astype(str):
        path = _matrix_path(source_dir, source_method, n_rois, file_id)
        if not path.exists():
            missing.append(str(path))
            continue
        matrix = np.asarray(np.load(path), dtype=float)
        if matrix.shape != (n_rois, n_rois):
            raise ValueError(f"Matriz con shape inesperado en {path}: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Matriz con NaN/inf en {path}")
        if public_method in DIRECTED_METHODS:
            if not np.allclose(np.diag(matrix), 0.0, atol=1e-6):
                raise ValueError(f"Matriz dirigida con diagonal no nula en {path}")
        elif not np.allclose(matrix, matrix.T, atol=1e-6):
            raise ValueError(f"Matriz asociativa no simetrica en {path}")
        matrices.append(matrix)

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Faltan {len(missing)} matrices para {public_method}. Primeros faltantes:\n{preview}"
        )

    return MethodData(
        method=public_method,
        source_method=source_method,
        matrices=np.asarray(matrices, dtype=float),
        directed=public_method in DIRECTED_METHODS,
        roi_names=_read_roi_names(source_dir, source_method, n_rois),
        manifest=manifest.reset_index(drop=True),
    )


def _save_matrix(matrix: np.ndarray, output_dir: Path, name: str, roi_names: Sequence[str]) -> None:
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    np.save(matrices_dir / f"{name}.npy", matrix)
    pd.DataFrame(matrix, index=roi_names, columns=roi_names).to_csv(matrices_dir / f"{name}.csv")


def _matrix_limits(*matrices: np.ndarray, symmetric: bool = True) -> tuple[float, float]:
    max_abs = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices)
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    if symmetric:
        return -max_abs, max_abs
    min_v = min(float(np.nanmin(matrix)) for matrix in matrices)
    max_v = max(float(np.nanmax(matrix)) for matrix in matrices)
    if min_v == max_v:
        return min_v - 1.0, max_v + 1.0
    return min_v, max_v


def _plot_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    roi_names: Sequence[str],
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    order, centers, network_labels, boundaries = _network_plot_layout(roi_names)
    ordered_matrix = matrix[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(
        ordered_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Red de destino")
    ax.set_ylabel("Red de origen")
    _set_network_ticks(ax, centers, network_labels, boundaries)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _network_plot_layout(
    roi_names: Sequence[str],
) -> tuple[np.ndarray, list[float], list[str], list[float]]:
    """Ordena las ROIs por red y devuelve centros y separadores legibles."""
    network_rank = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    networks = np.asarray([_roi_network(name) for name in roi_names], dtype=object)
    order = np.asarray(
        sorted(
            range(len(roi_names)),
            key=lambda idx: (network_rank.get(str(networks[idx]), 999), idx),
        ),
        dtype=int,
    )
    ordered_networks = networks[order]
    centers: list[float] = []
    labels: list[str] = []
    boundaries: list[float] = []
    start = 0
    while start < len(order):
        network = str(ordered_networks[start])
        end = start + 1
        while end < len(order) and str(ordered_networks[end]) == network:
            end += 1
        centers.append((start + end - 1) / 2)
        labels.append(NETWORK_PUBLIC_LABELS.get(network, network))
        if end < len(order):
            boundaries.append(end - 0.5)
        start = end
    return order, centers, labels, boundaries


def _set_network_ticks(
    ax,
    centers: Sequence[float],
    labels: Sequence[str],
    boundaries: Sequence[float],
) -> None:
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for boundary in boundaries:
        ax.axhline(boundary, color="#333333", linewidth=0.45, alpha=0.7)
        ax.axvline(boundary, color="#333333", linewidth=0.45, alpha=0.7)


def _plot_control_tea_panel(
    method: str,
    control_matrix: np.ndarray,
    tea_matrix: np.ndarray,
    diff_matrix: np.ndarray,
    output_dir: Path,
    roi_names: Sequence[str],
) -> Path:
    label = PUBLIC_LABELS.get(method, method)
    path = output_dir / "figures" / f"{method}_control_tea_diff_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    group_vmin, group_vmax = _matrix_limits(control_matrix, tea_matrix)
    diff_vmin, diff_vmax = _matrix_limits(diff_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=170)
    order, centers, network_labels, boundaries = _network_plot_layout(roi_names)
    panels = [
        ("Controles", control_matrix, group_vmin, group_vmax),
        ("TEA", tea_matrix, group_vmin, group_vmax),
        ("TEA - controles", diff_matrix, diff_vmin, diff_vmax),
    ]
    for ax, (title, matrix, vmin, vmax) in zip(axes, panels):
        ordered_matrix = matrix[np.ix_(order, order)]
        im = ax.imshow(
            ordered_matrix,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{label}: {title}")
        ax.set_xlabel("Red de destino")
        ax.set_ylabel("Red de origen")
        _set_network_ticks(ax, centers, network_labels, boundaries)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _connection_pairs(n_rois: int, directed: bool) -> Iterable[tuple[int, int]]:
    if directed:
        return ((i, j) for i in range(n_rois) for j in range(n_rois) if i != j)
    return ((i, j) for i in range(n_rois) for j in range(i + 1, n_rois))


def _rank_connections(
    method: str,
    control_matrix: np.ndarray,
    tea_matrix: np.ndarray,
    directed: bool,
    roi_names: Sequence[str],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    diff = tea_matrix - control_matrix
    for i, j in _connection_pairs(diff.shape[0], directed):
        value = float(diff[i, j])
        if value > 0:
            sign = "higher_in_TEA"
        elif value < 0:
            sign = "lower_in_TEA"
        else:
            sign = "no_difference"
        rows.append(
            {
                "method": method,
                "roi_i": roi_names[i],
                "roi_j": roi_names[j],
                "roi_i_idx": i,
                "roi_j_idx": j,
                "control_value": float(control_matrix[i, j]),
                "tea_value": float(tea_matrix[i, j]),
                "difference_tea_minus_control": value,
                "abs_difference": abs(value),
                "direction": "directed" if directed else "undirected",
                "sign_interpretation": sign,
            }
        )
    table = pd.DataFrame(rows).sort_values("abs_difference", ascending=False).reset_index(drop=True)
    table.insert(0, "rank_within_method", np.arange(1, len(table) + 1))
    return table.head(top_n).copy()


def _rank_group_connections(
    method: str,
    group: str,
    matrix: np.ndarray,
    directed: bool,
    roi_names: Sequence[str],
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for i, j in _connection_pairs(matrix.shape[0], directed):
        value = float(matrix[i, j])
        rows.append(
            {
                "method": method,
                "group": group,
                "roi_i": roi_names[i],
                "roi_j": roi_names[j],
                "roi_i_idx": i,
                "roi_j_idx": j,
                "connection_value": value,
                "abs_connection_value": abs(value),
                "direction": "directed" if directed else "undirected",
                "sign_interpretation": (
                    "positive_connection"
                    if value > 0
                    else "negative_connection"
                    if value < 0
                    else "zero_connection"
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("abs_connection_value", ascending=False).reset_index(drop=True)
    table.insert(0, "rank_within_group", np.arange(1, len(table) + 1))
    return table.head(top_n).copy()


def _top_matrix(diff_matrix: np.ndarray, top_table: pd.DataFrame, directed: bool) -> np.ndarray:
    out = np.zeros_like(diff_matrix, dtype=float)
    for _, row in top_table.iterrows():
        i = int(row["roi_i_idx"])
        j = int(row["roi_j_idx"])
        out[i, j] = diff_matrix[i, j]
        if not directed:
            out[j, i] = diff_matrix[j, i]
    return out


def _fallback_roi_coordinates(roi_names: Sequence[str]) -> np.ndarray:
    seen_by_network: Dict[str, int] = {}
    coords = []
    for roi_name in roi_names:
        hemi = _roi_hemisphere(str(roi_name))
        network = _roi_network(str(roi_name))
        count = seen_by_network.get(f"{hemi}_{network}", 0)
        seen_by_network[f"{hemi}_{network}"] = count + 1
        x = -45.0 if hemi == "LH" else 45.0 if hemi == "RH" else 0.0
        network_idx = NETWORK_ORDER.index(network) if network in NETWORK_ORDER else len(NETWORK_ORDER) - 1
        y = -70.0 + network_idx * 20.0
        z = -20.0 + (count % 8) * 8.0
        coords.append([x, y, z])
    return np.asarray(coords, dtype=float)


def _schaefer_mni_coordinates(n_rois: int, roi_names: Sequence[str]) -> tuple[np.ndarray, str]:
    if n_rois != 100:
        return _fallback_roi_coordinates(roi_names), "fallback_network_layout"

    try:
        import nibabel as nib
        from nilearn import datasets
    except ImportError:
        return _fallback_roi_coordinates(roi_names), "fallback_network_layout"

    try:
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=100,
            yeo_networks=7,
            resolution_mm=2,
            verbose=0,
        )
        img = nib.load(str(atlas.maps))
        data = np.rint(img.get_fdata()).astype(int)
        coords = []
        for rid in range(1, n_rois + 1):
            vox = np.argwhere(data == rid)
            if len(vox) == 0:
                coords.append([np.nan, np.nan, np.nan])
                continue
            centroid_vox = vox.mean(axis=0)
            centroid_world = nib.affines.apply_affine(img.affine, centroid_vox)
            coords.append(centroid_world)
        coords = np.asarray(coords, dtype=float)
        if np.isnan(coords).any():
            return _fallback_roi_coordinates(roi_names), "fallback_network_layout"
        return coords, "schaefer2018_mni_centroids"
    except Exception:
        return _fallback_roi_coordinates(roi_names), "fallback_network_layout"


def _adjacency_from_top_table(
    top_table: pd.DataFrame,
    n_rois: int,
    directed: bool,
    value_col: str = "difference_tea_minus_control",
) -> np.ndarray:
    adjacency = np.zeros((n_rois, n_rois), dtype=float)
    for _, row in top_table.iterrows():
        i = int(row["roi_i_idx"])
        j = int(row["roi_j_idx"])
        value = float(row[value_col])
        adjacency[i, j] = value
        # Nilearn glass-brain connectomes do not encode arrow direction.
        # For LiNGAM we mirror only for visualization; the CSV preserves direction.
        if directed:
            adjacency[j, i] = value
        else:
            adjacency[j, i] = value
    return adjacency


def _plot_top_connectome(
    method: str,
    top_table: pd.DataFrame,
    output_dir: Path,
    roi_names: Sequence[str],
    directed: bool,
    top_n: int = 20,
    value_col: str = "difference_tea_minus_control",
    filename: str | None = None,
    title: str | None = None,
    edge_limit: float | None = None,
) -> Path:
    try:
        from nilearn import plotting
    except ImportError as exc:
        raise ImportError("Los conectomas top 20 requieren nilearn.") from exc

    path = output_dir / "figures" / (filename or f"{method}_top_{top_n}_connectome.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    rank_col = "rank_within_method" if "rank_within_method" in top_table.columns else "rank_within_group"
    top = top_table.sort_values(rank_col).head(top_n).copy()
    coords, coord_source = _schaefer_mni_coordinates(len(roi_names), roi_names)
    adjacency = _adjacency_from_top_table(top, len(roi_names), directed, value_col=value_col)
    max_abs = edge_limit if edge_limit is not None else float(np.nanmax(np.abs(adjacency))) if np.any(adjacency) else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig = plt.figure(figsize=(7.6, 4.8), dpi=300)
    display = plotting.plot_connectome(
        adjacency,
        coords,
        edge_threshold=None,
        node_size=18,
        node_color="black",
        edge_cmap="coolwarm",
        edge_vmin=-max_abs,
        edge_vmax=max_abs,
        colorbar=True,
        figure=fig,
        title=None,
    )
    display.savefig(str(path), dpi=300)
    display.close()
    plt.close(fig)

    coord_path = output_dir / "tables" / "schaefer100_roi_coordinates.csv"
    if not coord_path.exists():
        pd.DataFrame(
            {
                "roi_idx": np.arange(len(roi_names)),
                "roi_name": list(roi_names),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "coordinate_source": coord_source,
            }
        ).to_csv(coord_path, index=False)
    return path


def _annotate_top_connections(top_table: pd.DataFrame) -> pd.DataFrame:
    annotated = top_table.copy()
    annotated["hemi_i"] = annotated["roi_i"].map(_roi_hemisphere)
    annotated["hemi_j"] = annotated["roi_j"].map(_roi_hemisphere)
    annotated["network_i"] = annotated["roi_i"].map(_roi_network)
    annotated["network_j"] = annotated["roi_j"].map(_roi_network)
    annotated["region_i"] = annotated["roi_i"].map(_roi_region)
    annotated["region_j"] = annotated["roi_j"].map(_roi_region)
    annotated["network_i_label"] = annotated["network_i"].map(
        lambda network: NETWORK_PUBLIC_LABELS.get(network, network)
    )
    annotated["network_j_label"] = annotated["network_j"].map(
        lambda network: NETWORK_PUBLIC_LABELS.get(network, network)
    )
    annotated["network_pair"] = [
        _network_pair(network_i, network_j)
        for network_i, network_j in zip(annotated["network_i"], annotated["network_j"])
    ]

    contexts = [
        _literature_context_for_pair(network_i, network_j)
        for network_i, network_j in zip(annotated["network_i"], annotated["network_j"])
    ]
    for column in ["literature_match_level", "literature_note", "citation_keys"]:
        annotated[column] = [context[column] for context in contexts]
    return annotated


def _plot_top20_network_summary(review: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "figures" / "top20_network_pairs_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = (
        review.groupby(["method", "network_pair"], as_index=False)
        .size()
        .sort_values(["method", "size"], ascending=[True, False])
    )
    pair_order = (
        counts.groupby("network_pair")["size"]
        .sum()
        .sort_values(ascending=False)
        .head(14)
        .index
        .tolist()
    )
    plot_data = counts[counts["network_pair"].isin(pair_order)].pivot_table(
        index="network_pair",
        columns="method",
        values="size",
        fill_value=0,
        aggfunc="sum",
    )
    plot_data = plot_data.loc[pair_order]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=170)
    left = np.zeros(len(plot_data), dtype=float)
    colors = {
        "correlacion": "#4c78a8",
        "lasso": "#f58518",
        "lingam": "#54a24b",
    }
    y = np.arange(len(plot_data))
    for method in [method for method in ["correlacion", "lasso", "lingam"] if method in plot_data.columns]:
        values = plot_data[method].to_numpy(dtype=float)
        ax.barh(y, values, left=left, label=method, color=colors.get(method))
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data.index)
    ax.invert_yaxis()
    ax.set_xlabel("Numero de conexiones en top 20")
    ax.set_title("Pares de redes mas frecuentes en top 20 por metodo")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_connectome_triplet_panel(
    method: str,
    control_path: Path,
    tea_path: Path,
    diff_path: Path,
    output_dir: Path,
) -> Path:
    path = output_dir / "figures" / f"{method}_control_tea_diff_top20_connectomes_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22, 7.2),
        dpi=300,
        gridspec_kw={"wspace": 0.18},
    )
    panels = [
        ("Controles", control_path),
        ("TEA", tea_path),
        ("TEA − controles", diff_path),
    ]
    for ax, (title, image_path) in zip(axes, panels):
        ax.imshow(plt.imread(image_path))
        ax.set_title(title, fontsize=16, pad=16)
        ax.axis("off")
    fig.suptitle(
        CONNECTOME_PRESENTATION_TITLES.get(method, f"Conectomas promedio y diferencias entre grupos — {method}"),
        fontsize=20,
        y=0.96,
    )
    note_lines = [
        "En TEA − controles, los valores positivos indican mayor conectividad en TEA y los negativos mayor conectividad en controles."
    ]
    if method == "lingam":
        note_lines.append("En LiNGAM, la dirección de las conexiones se conserva en el archivo CSV correspondiente.")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.84, bottom=0.16, wspace=0.18)
    fig.text(
        0.5,
        0.055,
        "\n".join(note_lines),
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def build_top20_literature_outputs(top_tables: Sequence[pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    review = pd.concat([_annotate_top_connections(table) for table in top_tables], ignore_index=True)
    review = review.sort_values(["method", "rank_within_method"]).reset_index(drop=True)
    review_path = tables_dir / "top20_connections_literature_review.csv"
    review.to_csv(review_path, index=False)

    network_summary = (
        review.groupby(["method", "network_pair", "literature_match_level"], as_index=False)
        .size()
        .sort_values(["method", "size"], ascending=[True, False])
        .rename(columns={"size": "n_top20_connections"})
    )
    summary_path = tables_dir / "top20_network_summary.csv"
    network_summary.to_csv(summary_path, index=False)

    references_path = tables_dir / "literature_references.csv"
    pd.DataFrame(LITERATURE_REFERENCES).to_csv(references_path, index=False)

    summary_fig = _plot_top20_network_summary(review, output_dir)
    report_path = _write_top20_literature_report(review, network_summary, output_dir)
    return {
        "top20_literature_review": str(review_path),
        "top20_network_summary": str(summary_path),
        "literature_references": str(references_path),
        "top20_network_summary_fig": str(summary_fig),
        "top20_literature_report": str(report_path),
    }


def _write_top20_literature_report(
    review: pd.DataFrame,
    network_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    lines = [
        "# Comparacion top 20 con literatura TEA",
        "",
        "Esta salida cruza las conexiones top 20 de cada metodo con las redes Schaefer-100/Yeo-7. "
        "Es una lectura de consistencia anatomico-funcional con literatura, no una revision sistematica "
        "ni una validacion clinica independiente.",
        "",
        "## Fuentes usadas como marco",
        "",
    ]
    for reference in LITERATURE_REFERENCES:
        lines.append(
            f"- {reference['citation_key']}: {reference['short_reference']}. "
            f"{reference['relevance']} URL: {reference['url']}"
        )

    lines.extend(["", "## Resumen por metodo", ""])
    for method in [m for m in ["correlacion", "lasso", "lingam"] if m in set(review["method"])]:
        method_summary = network_summary[network_summary["method"].eq(method)].head(6)
        method_review = review[review["method"].eq(method)]
        high = int(method_review["literature_match_level"].eq("high").sum())
        moderate = int(method_review["literature_match_level"].eq("moderate").sum())
        exploratory = int(method_review["literature_match_level"].eq("exploratory").sum())
        lines.extend(
            [
                f"### {method}",
                "",
                f"- Top 20: coincidencia alta={high}, moderada={moderate}, exploratoria={exploratory}.",
                "- Pares de redes mas frecuentes:",
                "",
                "| Par de redes | Nivel literatura | n |",
                "|---|---:|---:|",
            ]
        )
        for _, row in method_summary.iterrows():
            lines.append(
                f"| {row['network_pair']} | {row['literature_match_level']} | "
                f"{int(row['n_top20_connections'])} |"
            )
        best = method_review.sort_values("abs_difference", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                (
                    f"Conexion maxima: `{best['roi_i']}` -> `{best['roi_j']}` "
                    f"({best['network_pair']}), TEA − controles={_fmt(best['difference_tea_minus_control'])}."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Lectura sugerida",
            "",
            (
                "Si las top 20 se concentran en Default, saliencia/atencion ventral, control/frontoparietal "
                "o atencion dorsal, se puede decir que las diferencias de esta cohorte caen en sistemas "
                "frecuentemente discutidos en conectividad funcional TEA. No conviene afirmar que una conexion "
                "individual replica un paper especifico salvo que se haga una revision dirigida ROI por ROI."
            ),
            "",
            (
                "Para LiNGAM, el conectoma muestra la ubicacion espacial de las aristas top 20; la direccion "
                "causal estimada debe leerse desde `top20_connections_literature_review.csv` y no desde la figura."
            ),
            "",
        ]
    )
    path = output_dir / "top20_literature_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _support_metrics(
    method: str,
    group_name: str,
    matrix: np.ndarray,
    threshold: float,
    n_subjects: int,
    directed: bool,
) -> Dict[str, object]:
    n = matrix.shape[0]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    if not directed:
        mask = np.triu(mask, k=1)
    values = matrix[mask]
    nonzero = np.abs(values) > threshold
    return {
        "method": method,
        "group": group_name,
        "threshold": threshold,
        "n_subjects": n_subjects,
        "mean_abs_magnitude_group_matrix": float(np.mean(np.abs(values))) if values.size else np.nan,
        "sparsity_group_matrix": float(1.0 - np.mean(nonzero)) if values.size else np.nan,
        "n_nonzero": int(nonzero.sum()),
        "n_zero_like": int((~nonzero).sum()),
        "n_total_connections": int(values.size),
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _format_connection_count(nonzero: int, total: int) -> str:
    return f"{int(nonzero)} / {int(total)}"


def _choose_support_threshold(support: pd.DataFrame, preferred: float) -> float:
    thresholds = np.asarray(sorted(support["threshold"].astype(float).unique()), dtype=float)
    if thresholds.size == 0:
        raise ValueError("No hay umbrales disponibles para construir la tabla de esparsidad.")
    matches = thresholds[np.isclose(thresholds, preferred)]
    if matches.size:
        return float(matches[0])
    return float(thresholds[-1])


def _build_sparsity_support_table(support: pd.DataFrame, threshold: float) -> pd.DataFrame:
    support_ref = support[np.isclose(support["threshold"].astype(float), threshold)].copy()
    rows: list[dict[str, object]] = []
    method_order = [method for method in METHOD_ALIASES if method in set(support_ref["method"].astype(str))]

    for method in method_order:
        method_rows = support_ref[support_ref["method"].eq(method)].set_index("group")
        if not {"Controles", "TEA"}.issubset(method_rows.index):
            continue
        control = method_rows.loc["Controles"]
        tea = method_rows.loc["TEA"]
        total_control = int(control["n_total_connections"])
        total_tea = int(tea["n_total_connections"])
        total_display = total_control if total_control == total_tea else f"{total_control} / {total_tea}"
        rows.append(
            {
                "Método": SPARSITY_SUPPORT_LABELS.get(method, method),
                "Tipo de red": "Dirigida" if method in DIRECTED_METHODS else "No dirigida",
                "Umbral": f"|peso| < {threshold:g}",
                "Esparsidad controles": _format_percent(float(control["sparsity_group_matrix"])),
                "Esparsidad TEA": _format_percent(float(tea["sparsity_group_matrix"])),
                "Conexiones activas controles": _format_connection_count(
                    int(control["n_nonzero"]), total_control
                ),
                "Conexiones activas TEA": _format_connection_count(int(tea["n_nonzero"]), total_tea),
                "Total conexiones evaluadas": total_display,
                "Lectura para tesis": SPARSITY_SUPPORT_NOTES.get(method, ""),
            }
        )

    return pd.DataFrame(rows)


def _write_sparsity_support_markdown(table: pd.DataFrame, path: Path, threshold: float) -> Path:
    headers = list(table.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in headers]
        markdown_rows.append("| " + " | ".join(values) + " |")

    lines = [
        "# Tabla de apoyo. Esparsidad de matrices promedio por método",
        "",
        f"Esparsidad calculada como proporción de conexiones con peso cercano a cero usando `{threshold:g}`.",
        "",
        *markdown_rows,
        "",
        (
            "Nota: la esparsidad se interpreta principalmente como propiedad del método de estimación "
            "y del umbral aplicado, no como evidencia de que un grupo tenga redes globalmente más "
            "o menos dispersas."
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_sparsity_support_table(table: pd.DataFrame, path: Path, threshold: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_table = table[
        [
            "Método",
            "Tipo de red",
            "Esparsidad controles",
            "Esparsidad TEA",
            "Conexiones activas controles",
            "Conexiones activas TEA",
            "Lectura para tesis",
        ]
    ].copy()
    plot_table = plot_table.rename(
        columns={
            "Conexiones activas controles": "Controles activas",
            "Conexiones activas TEA": "TEA activas",
            "Lectura para tesis": "Lectura",
        }
    )
    plot_table["Controles activas"] = plot_table["Controles activas"].map(
        lambda value: str(value).replace(" / ", "\n/ ")
    )
    plot_table["TEA activas"] = plot_table["TEA activas"].map(
        lambda value: str(value).replace(" / ", "\n/ ")
    )
    plot_table["Lectura"] = plot_table["Lectura"].map(
        lambda value: "\n".join(textwrap.wrap(str(value), width=34))
    )

    fig, ax = plt.subplots(figsize=(15.5, 3.8), dpi=300)
    ax.axis("off")
    ax.set_title(
        "Tabla de apoyo. Esparsidad de matrices promedio por método",
        fontsize=15,
        fontweight="bold",
        pad=10,
    )

    col_widths = [0.17, 0.11, 0.11, 0.10, 0.14, 0.13, 0.24]
    table_artist = ax.table(
        cellText=plot_table.values,
        colLabels=plot_table.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.25, 1.0, 0.58],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9.2)
    table_artist.scale(1.0, 2.0)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#c8d0d9")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#273444")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f7f9fb" if row % 2 else "#ffffff")
            if col in {0, 6}:
                cell.get_text().set_ha("left")

    fig.text(
        0.02,
        0.035,
        (
            f"Nota: esparsidad = proporción de conexiones con |peso| menor o igual a {threshold:g}. "
            "TEA y controles son similares dentro de cada método; la diferencia principal es metodológica."
        ),
        fontsize=9,
        color="#2f3b47",
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def build_sparsity_support_outputs(
    support: pd.DataFrame,
    output_dir: Path,
    preferred_threshold: float = SPARSITY_SUPPORT_THRESHOLD,
) -> Dict[str, str]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    threshold = _choose_support_threshold(support, preferred_threshold)
    table = _build_sparsity_support_table(support, threshold)
    csv_path = tables_dir / "sparsity_support_table.csv"
    md_path = tables_dir / "sparsity_support_table.md"
    png_path = figures_dir / "sparsity_support_table.png"

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_sparsity_support_markdown(table, md_path, threshold)
    _plot_sparsity_support_table(table, png_path, threshold)

    return {
        "sparsity_support_table": str(csv_path),
        "sparsity_support_table_md": str(md_path),
        "sparsity_support_table_fig": str(png_path),
    }


def _read_graphical_lasso_alpha(source_dir: Path) -> float | None:
    config_path = source_dir / "configuracion_pipeline.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            alpha = config.get("graphical_lasso_alpha")
            if alpha is not None:
                return float(alpha)
        except Exception:
            pass

    matrix_root = source_dir / "matrices"
    for meta_path in matrix_root.glob("graphical_lasso_*rois/*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            alpha = meta.get("graphical_lasso_alpha")
            if alpha is not None:
                return float(alpha)
        except Exception:
            continue
    return None


def build_method_metadata(output_dir: Path, source_dir: Path | None = None) -> Path:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    alpha = _read_graphical_lasso_alpha(source_dir) if source_dir is not None else None
    alpha_status = f"fixed_alpha={alpha:g}" if alpha is not None else "not_available_in_existing_cache"
    alpha_note = (
        f"Alpha fijo validado y usado en la cohorte: {alpha:g}."
        if alpha is not None
        else "No se recupera alpha por sujeto desde caches antiguos."
    )
    rows = [
        {
            "method": "lasso",
            "implementation_name": "graphical_lasso",
            "public_label": "Graphical Lasso",
            "stored_matrix": "normalized_partial_correlation_from_precision",
            "is_raw_precision_matrix": False,
            "is_sparse_expected": True,
            "alpha_status": alpha_status,
            "note": (
                "La matriz cacheada se interpreta como forma normalizada tipo correlacion parcial "
                f"derivada de la precision estimada; {alpha_note}"
            ),
        }
    ]
    path = tables_dir / "lasso_method_metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def generate_interpretation_report(source_dir: str | Path, output_dir: str | Path) -> Path:
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    tables_dir = output_dir / "tables"
    top_path = tables_dir / "top_different_connections.csv"
    support_path = tables_dir / "support_metrics_summary.csv"
    lasso_meta_path = tables_dir / "lasso_method_metadata.csv"

    missing = [path for path in [top_path, support_path, lasso_meta_path] if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Faltan tablas para generar la interpretacion:\n{preview}")

    top = pd.read_csv(top_path)
    support = pd.read_csv(support_path)
    lasso_meta = pd.read_csv(lasso_meta_path)

    cohort = support.groupby("group")["n_subjects"].max().to_dict()
    methods = list(dict.fromkeys(top["method"].astype(str)))
    threshold = float(support["threshold"].max()) if not support.empty else np.nan
    support_ref = support[np.isclose(support["threshold"].astype(float), threshold)].copy()

    lines = [
        "# Interpretacion de resultados",
        "",
        "## Contexto",
        "",
        f"- Corrida base: `{source_dir}`",
        f"- Salida v9: `{output_dir}`",
        f"- Cohorte: controles={int(cohort.get('Controles', 0))}, TEA={int(cohort.get('TEA', 0))}.",
        f"- Metodos analizados: {', '.join(methods)}.",
        "",
        "Esta interpretacion resume diferencias descriptivas entre grupos. No debe leerse como prueba causal clinica ni como biomarcador validado independiente.",
        "Las matrices LiNGAM fueron normalizadas por el mayor coeficiente absoluto de cada sujeto; sus valores son pesos relativos y no coeficientes causales crudos.",
        "",
        "## Lectura por metodo",
        "",
    ]

    for method in methods:
        method_top = top[top["method"].eq(method)].copy()
        if method_top.empty:
            continue
        best = method_top.sort_values("abs_difference", ascending=False).iloc[0]
        counts = method_top["sign_interpretation"].value_counts().to_dict()
        higher = int(counts.get("higher_in_TEA", 0))
        lower = int(counts.get("lower_in_TEA", 0))
        direction = str(best.get("direction", ""))
        lines.extend(
            [
                f"### {method}",
                "",
                f"- Conexion con mayor diferencia absoluta: `{best['roi_i']}` -> `{best['roi_j']}`.",
                f"- Valor controles={_fmt(best['control_value'])}, TEA={_fmt(best['tea_value'])}, TEA - controles={_fmt(best['difference_tea_minus_control'])}.",
                f"- En el top {len(method_top)}, conexiones mayores en TEA={higher} y menores en TEA={lower}.",
                f"- Tipo de matriz: {direction}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Magnitud y soporte",
            "",
            f"Resumen usando umbral {threshold:g} para contar conexiones no nulas en matrices promedio.",
            "",
            "| Método | Grupo | Magnitud media abs. | Esparsidad | Conexiones no nulas | Total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in support_ref.sort_values(["method", "group"]).iterrows():
        lines.append(
            "| "
            f"{row['method']} | {row['group']} | "
            f"{_fmt(row['mean_abs_magnitude_group_matrix'])} | "
            f"{_fmt(row['sparsity_group_matrix'])} | "
            f"{int(row['n_nonzero'])} | {int(row['n_total_connections'])} |"
        )

    if not lasso_meta.empty:
        meta = lasso_meta.iloc[0]
        lines.extend(
            [
                "",
                "## Nota metodológica sobre Graphical Lasso",
                "",
                f"- Implementacion interna: `{meta.get('implementation_name', 'graphical_lasso')}`.",
                f"- Matriz guardada: `{meta.get('stored_matrix', 'normalized_partial_correlation_from_precision')}`.",
                f"- Alpha: `{meta.get('alpha_status', 'NA')}`.",
                f"- Nota: {meta.get('note', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Frase sugerida para la tesis",
            "",
            (
                "Las matrices promedio permiten comparar descriptivamente TEA y controles, "
                "pero la interpretación debe mantenerse como evidencia empírica de la cohorte. "
                "Pearson resume conectividad funcional clásica, Graphical Lasso aproxima asociaciones "
                "condicionadas mediante precisión regularizada y LiNGAM se reporta como contraste "
                "direccional exploratorio."
            ),
            "",
        ]
    )

    path = output_dir / "interpretacion_resultados.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _clean_v9_outputs(output_dir: Path) -> None:
    for folder in ["figures", "tables", "matrices"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    for pattern in ["partial_*", "pearson_*", "graphical_lasso_*"]:
        for path in output_dir.rglob(pattern):
            if path.is_file():
                path.unlink()


def run_group_connectivity_analysis(
    source_dir: str | Path,
    output_dir: str | Path,
    methods: Sequence[str] = ("correlacion", "lasso", "lingam"),
    thresholds: Sequence[float] = (1e-5, 1e-4, 1e-3),
    top_n_edges: int = 100,
) -> Dict[str, object]:
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    _clean_v9_outputs(output_dir)

    manifest = load_subject_manifest(source_dir)
    all_top_tables = []
    top20_tables = []
    group_top20_tables = []
    support_rows = []
    outputs: Dict[str, object] = {"methods": {}}

    for method in methods:
        data = load_method_data(source_dir, method, manifest)
        control_mask = data.manifest["group"].eq("Control").to_numpy()
        tea_mask = data.manifest["group"].eq("TEA").to_numpy()
        if not control_mask.any() or not tea_mask.any():
            raise ValueError(f"La cohorte para {method} no contiene sujetos de TEA y controles.")

        control_matrix = np.mean(data.matrices[control_mask], axis=0)
        tea_matrix = np.mean(data.matrices[tea_mask], axis=0)
        diff_matrix = tea_matrix - control_matrix
        abs_diff_matrix = np.abs(tea_matrix) - np.abs(control_matrix)

        _save_matrix(control_matrix, output_dir, f"{method}_control_mean", data.roi_names)
        _save_matrix(tea_matrix, output_dir, f"{method}_tea_mean", data.roi_names)
        _save_matrix(diff_matrix, output_dir, f"{method}_tea_minus_control", data.roi_names)
        _save_matrix(abs_diff_matrix, output_dir, f"{method}_abs_diff_tea_minus_control", data.roi_names)

        panel = _plot_control_tea_panel(
            method, control_matrix, tea_matrix, diff_matrix, output_dir, data.roi_names
        )
        abs_vmin, abs_vmax = _matrix_limits(abs_diff_matrix)
        abs_fig = _plot_heatmap(
            abs_diff_matrix,
            output_dir / "figures" / f"{method}_abs_diff_tea_minus_control_heatmap.png",
            f"{PUBLIC_LABELS.get(method, method)}: |TEA| - |controles|",
            data.roi_names,
            vmin=abs_vmin,
            vmax=abs_vmax,
        )

        control_top20_group = _rank_group_connections(
            method, "Controles", control_matrix, data.directed, data.roi_names, 20
        )
        tea_top20_group = _rank_group_connections(
            method, "TEA", tea_matrix, data.directed, data.roi_names, 20
        )
        group_top20_tables.extend([control_top20_group, tea_top20_group])
        control_top20_group_path = output_dir / "tables" / f"{method}_control_top_20_connections.csv"
        tea_top20_group_path = output_dir / "tables" / f"{method}_tea_top_20_connections.csv"
        control_top20_group.to_csv(control_top20_group_path, index=False)
        tea_top20_group.to_csv(tea_top20_group_path, index=False)
        group_edge_limit = max(
            float(control_top20_group["abs_connection_value"].max()),
            float(tea_top20_group["abs_connection_value"].max()),
        )
        control_top20_connectome = _plot_top_connectome(
            method,
            control_top20_group,
            output_dir,
            data.roi_names,
            data.directed,
            top_n=20,
            value_col="connection_value",
            filename=f"{method}_control_top_20_connectome.png",
            title=f"{PUBLIC_LABELS.get(method, method)}: controles, 20 conexiones",
            edge_limit=group_edge_limit,
        )
        tea_top20_connectome = _plot_top_connectome(
            method,
            tea_top20_group,
            output_dir,
            data.roi_names,
            data.directed,
            top_n=20,
            value_col="connection_value",
            filename=f"{method}_tea_top_20_connectome.png",
            title=f"{PUBLIC_LABELS.get(method, method)}: TEA, 20 conexiones",
            edge_limit=group_edge_limit,
        )

        top_table = _rank_connections(
            method, control_matrix, tea_matrix, data.directed, data.roi_names, top_n_edges
        )
        all_top_tables.append(top_table)
        method_top_path = output_dir / "tables" / f"{method}_top_{top_n_edges}_different_connections.csv"
        top_table.to_csv(method_top_path, index=False)

        top_matrix = _top_matrix(diff_matrix, top_table, data.directed)
        top_vmin, top_vmax = _matrix_limits(top_matrix)
        top_fig = _plot_heatmap(
            top_matrix,
            output_dir / "figures" / f"{method}_top_{top_n_edges}_different_connections_heatmap.png",
            f"{PUBLIC_LABELS.get(method, method)}: top {top_n_edges} diferencias",
            data.roi_names,
            vmin=top_vmin,
            vmax=top_vmax,
        )

        top20_table = _rank_connections(
            method, control_matrix, tea_matrix, data.directed, data.roi_names, 20
        )
        top20_tables.append(top20_table)
        method_top20_path = output_dir / "tables" / f"{method}_top_20_different_connections.csv"
        top20_table.to_csv(method_top20_path, index=False)
        top20_matrix = _top_matrix(diff_matrix, top20_table, data.directed)
        top20_vmin, top20_vmax = _matrix_limits(top20_matrix)
        top20_fig = _plot_heatmap(
            top20_matrix,
            output_dir / "figures" / f"{method}_top_20_different_connections_heatmap.png",
            f"{PUBLIC_LABELS.get(method, method)}: top 20 diferencias",
            data.roi_names,
            vmin=top20_vmin,
            vmax=top20_vmax,
        )
        top20_connectome = _plot_top_connectome(
            method,
            top20_table,
            output_dir,
            data.roi_names,
            data.directed,
            top_n=20,
        )
        top20_connectome_panel = _plot_connectome_triplet_panel(
            method,
            control_top20_connectome,
            tea_top20_connectome,
            top20_connectome,
            output_dir,
        )

        if method == "lasso":
            for n in [20, 50, 100]:
                lasso_top = _rank_connections(
                    method, control_matrix, tea_matrix, data.directed, data.roi_names, n
                )
                lasso_top.to_csv(output_dir / "tables" / f"lasso_top_{n}_different_connections.csv", index=False)
                lasso_top_matrix = _top_matrix(diff_matrix, lasso_top, data.directed)
                lasso_vmin, lasso_vmax = _matrix_limits(lasso_top_matrix)
                _plot_heatmap(
                    lasso_top_matrix,
                    output_dir / "figures" / f"lasso_top_{n}_different_connections_heatmap.png",
                    f"Graphical Lasso: {n} mayores diferencias",
                    data.roi_names,
                    vmin=lasso_vmin,
                    vmax=lasso_vmax,
                )

        for threshold in thresholds:
            support_rows.append(
                _support_metrics(method, "Controles", control_matrix, threshold, int(control_mask.sum()), data.directed)
            )
            support_rows.append(
                _support_metrics(method, "TEA", tea_matrix, threshold, int(tea_mask.sum()), data.directed)
            )

        outputs["methods"][method] = {
            "panel": str(panel),
            "abs_diff_heatmap": str(abs_fig),
            "top_heatmap": str(top_fig),
            "top_table": str(method_top_path),
            "top20_table": str(method_top20_path),
            "top20_heatmap": str(top20_fig),
            "top20_connectome": str(top20_connectome),
            "control_top20_table": str(control_top20_group_path),
            "tea_top20_table": str(tea_top20_group_path),
            "control_top20_connectome": str(control_top20_connectome),
            "tea_top20_connectome": str(tea_top20_connectome),
            "top20_connectome_panel": str(top20_connectome_panel),
            "n_control": int(control_mask.sum()),
            "n_tea": int(tea_mask.sum()),
            "n_rois": len(data.roi_names),
        }

    top_all = pd.concat(all_top_tables, ignore_index=True)
    top_all = top_all.sort_values("abs_difference", ascending=False).reset_index(drop=True)
    top_all.insert(0, "rank_overall", np.arange(1, len(top_all) + 1))
    top_all.to_csv(output_dir / "tables" / "top_different_connections.csv", index=False)

    support_summary = pd.DataFrame(support_rows)
    support_summary_path = output_dir / "tables" / "support_metrics_summary.csv"
    support_summary.to_csv(support_summary_path, index=False)
    sparsity_support_outputs = build_sparsity_support_outputs(support_summary, output_dir)
    top20_group_all = pd.concat(group_top20_tables, ignore_index=True)
    top20_group_all = _annotate_top_connections(top20_group_all)
    top20_group_all.to_csv(output_dir / "tables" / "top20_group_connections.csv", index=False)
    metadata_path = build_method_metadata(output_dir, source_dir=source_dir)
    interpretation_path = generate_interpretation_report(source_dir, output_dir)
    top20_outputs = build_top20_literature_outputs(top20_tables, output_dir)
    outputs["top_connections"] = str(output_dir / "tables" / "top_different_connections.csv")
    outputs["support_metrics"] = str(support_summary_path)
    outputs["lasso_metadata"] = str(metadata_path)
    outputs["interpretation_report"] = str(interpretation_path)
    outputs.update(sparsity_support_outputs)
    outputs.update(top20_outputs)
    return outputs
