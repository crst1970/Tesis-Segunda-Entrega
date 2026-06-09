from __future__ import annotations

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
    "correlacion": "Correlacion",
    "lasso": "Lasso",
    "lingam": "LiNGAM",
}

DIRECTED_METHODS = {"lingam"}


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
    manifest["group"] = np.where(manifest["label"] == 1, "TEA", "Control")

    n_rois = manifest["n_rois"].dropna().astype(int).unique()
    if len(n_rois) != 1:
        raise ValueError(f"La cohorte mezcla cantidades de ROIs: {sorted(n_rois)}")

    return manifest


def _read_roi_names(source_dir: Path, source_method: str, n_rois: int) -> List[str]:
    feature_path = source_dir / "feature_maps" / f"{source_method}_feature_map.csv"
    if not feature_path.exists():
        return [f"ROI_{i + 1:03d}" for i in range(n_rois)]

    fmap = pd.read_csv(feature_path)
    required = {"roi_origen_idx", "roi_origen"}
    if not required.issubset(fmap.columns):
        return [f"ROI_{i + 1:03d}" for i in range(n_rois)]

    names = [f"ROI_{i + 1:03d}" for i in range(n_rois)]
    first_by_idx = fmap.drop_duplicates("roi_origen_idx")
    for _, row in first_by_idx.iterrows():
        idx = int(row["roi_origen_idx"])
        if 0 <= idx < n_rois:
            names[idx] = str(row["roi_origen"])
    return names


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
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("ROI destino")
    ax.set_ylabel("ROI origen")
    _thin_roi_ticks(ax, roi_names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _thin_roi_ticks(ax, roi_names: Sequence[str]) -> None:
    n = len(roi_names)
    if n <= 20:
        ticks = np.arange(n)
    else:
        step = max(1, int(np.ceil(n / 12)))
        ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([roi_names[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([roi_names[i] for i in ticks], fontsize=6)


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
    panels = [
        ("Control", control_matrix, group_vmin, group_vmax),
        ("TEA", tea_matrix, group_vmin, group_vmax),
        ("TEA - Control", diff_matrix, diff_vmin, diff_vmax),
    ]
    for ax, (title, matrix, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{label}: {title}")
        ax.set_xlabel("ROI destino")
        ax.set_ylabel("ROI origen")
        _thin_roi_ticks(ax, roi_names)
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


def _top_matrix(diff_matrix: np.ndarray, top_table: pd.DataFrame, directed: bool) -> np.ndarray:
    out = np.zeros_like(diff_matrix, dtype=float)
    for _, row in top_table.iterrows():
        i = int(row["roi_i_idx"])
        j = int(row["roi_j_idx"])
        out[i, j] = diff_matrix[i, j]
        if not directed:
            out[j, i] = diff_matrix[j, i]
    return out


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


def build_method_metadata(output_dir: Path) -> Path:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": "lasso",
            "implementation_name": "graphical_lasso",
            "public_label": "Lasso",
            "stored_matrix": "normalized_partial_correlation_from_precision",
            "is_raw_precision_matrix": False,
            "is_sparse_expected": True,
            "alpha_status": "not_available_in_existing_cache",
            "note": (
                "La matriz cacheada se interpreta como forma normalizada tipo correlacion parcial "
                "derivada de la precision estimada; no se recupera alpha por sujeto desde caches antiguos."
            ),
        }
    ]
    path = tables_dir / "lasso_method_metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
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
    support_rows = []
    outputs: Dict[str, object] = {"methods": {}}

    for method in methods:
        data = load_method_data(source_dir, method, manifest)
        control_mask = data.manifest["group"].eq("Control").to_numpy()
        tea_mask = data.manifest["group"].eq("TEA").to_numpy()
        if not control_mask.any() or not tea_mask.any():
            raise ValueError(f"La cohorte para {method} no tiene ambos grupos Control y TEA.")

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
            f"{PUBLIC_LABELS.get(method, method)}: abs(TEA) - abs(Control)",
            data.roi_names,
            vmin=abs_vmin,
            vmax=abs_vmax,
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
                    f"Lasso: top {n} diferencias",
                    data.roi_names,
                    vmin=lasso_vmin,
                    vmax=lasso_vmax,
                )

        for threshold in thresholds:
            support_rows.append(
                _support_metrics(method, "Control", control_matrix, threshold, int(control_mask.sum()), data.directed)
            )
            support_rows.append(
                _support_metrics(method, "TEA", tea_matrix, threshold, int(tea_mask.sum()), data.directed)
            )

        outputs["methods"][method] = {
            "panel": str(panel),
            "abs_diff_heatmap": str(abs_fig),
            "top_heatmap": str(top_fig),
            "top_table": str(method_top_path),
            "n_control": int(control_mask.sum()),
            "n_tea": int(tea_mask.sum()),
            "n_rois": len(data.roi_names),
        }

    top_all = pd.concat(all_top_tables, ignore_index=True)
    top_all = top_all.sort_values("abs_difference", ascending=False).reset_index(drop=True)
    top_all.insert(0, "rank_overall", np.arange(1, len(top_all) + 1))
    top_all.to_csv(output_dir / "tables" / "top_different_connections.csv", index=False)

    pd.DataFrame(support_rows).to_csv(output_dir / "tables" / "support_metrics_summary.csv", index=False)
    metadata_path = build_method_metadata(output_dir)
    outputs["top_connections"] = str(output_dir / "tables" / "top_different_connections.csv")
    outputs["support_metrics"] = str(output_dir / "tables" / "support_metrics_summary.csv")
    outputs["lasso_metadata"] = str(metadata_path)
    return outputs
