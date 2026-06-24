"""Analisis estadistico derivado de la corrida principal ABIDE I.

Este modulo no recalcula matrices de conectividad. Usa las matrices individuales
ya validadas para:

- describir covariables demograficas y movimiento;
- ajustar comparaciones TEA y controles por edad, sexo, movimiento y sitio;
- controlar comparaciones multiples mediante FDR de Benjamini-Hochberg;
- resumir magnitud y esparsidad por sujeto;
- calcular intervalos de confianza de metricas de clasificacion;
- regenerar figuras con terminologia academica en espanol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.multitest import fdrcorrection

try:
    from .analisis_control_tea import (
        DIRECTED_METHODS,
        NETWORK_ORDER,
        NETWORK_PUBLIC_LABELS,
        PUBLIC_LABELS,
        _roi_network,
        load_method_data,
        load_subject_manifest,
    )
except ImportError:
    from analisis_control_tea import (
        DIRECTED_METHODS,
        NETWORK_ORDER,
        NETWORK_PUBLIC_LABELS,
        PUBLIC_LABELS,
        _roi_network,
        load_method_data,
        load_subject_manifest,
    )


PRIMARY_METHODS = ("correlacion", "lasso")
ALL_METHODS = ("correlacion", "lasso", "lingam")
SPARSITY_THRESHOLD = 1e-3


@dataclass
class LinearModelResult:
    beta: np.ndarray
    standard_error: np.ndarray
    statistic: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    partial_r: np.ndarray
    degrees_freedom: int


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _clean_numeric(series: pd.Series, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(lower, upper))


def load_covariates(
    source_dir: str | Path,
    phenotype_csv: str | Path,
) -> pd.DataFrame:
    """Combina la cohorte final con covariables fenotipicas y de movimiento."""
    source_dir = _as_path(source_dir)
    phenotype_csv = _as_path(phenotype_csv)
    manifest = load_subject_manifest(source_dir).copy()
    phenotype = pd.read_csv(phenotype_csv, low_memory=False).copy()

    manifest["subject_key"] = pd.to_numeric(manifest["subject_id"], errors="coerce").astype("Int64")
    phenotype["subject_key"] = pd.to_numeric(phenotype["SUB_ID"], errors="coerce").astype("Int64")
    merged = manifest.merge(phenotype, on="subject_key", how="left", validate="one_to_one")

    if merged["DX_GROUP"].isna().any():
        missing = merged.loc[merged["DX_GROUP"].isna(), "file_id"].astype(str).tolist()
        raise ValueError(f"Faltan covariables para {len(missing)} sujetos: {missing[:5]}")

    merged["grupo"] = merged["group"].map({"TEA": "TEA", "Control": "Controles"})
    merged["diagnostico_tea"] = merged["group"].eq("TEA").astype(int)
    merged["edad"] = _clean_numeric(merged["AGE_AT_SCAN"], 4, 70)
    merged["sexo_femenino"] = pd.to_numeric(merged["SEX"], errors="coerce").map({1: 0.0, 2: 1.0})
    merged["movimiento_fd_medio"] = _clean_numeric(merged["func_mean_fd"], 0, 5)
    merged["porcentaje_fd"] = _clean_numeric(merged["func_perc_fd"], 0, 100)
    merged["ci_total"] = _clean_numeric(merged["FIQ"], 40, 160)
    merged["sitio"] = merged["site"].astype(str)

    keep = [
        "subject_id",
        "file_id",
        "group",
        "grupo",
        "diagnostico_tea",
        "sitio",
        "edad",
        "sexo_femenino",
        "movimiento_fd_medio",
        "porcentaje_fd",
        "ci_total",
    ]
    return merged[keep].reset_index(drop=True)


def _mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.mean():.2f} ({values.std(ddof=1):.2f})"


def build_covariate_summary(covariates: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla compacta para el manuscrito."""
    rows = []
    for group_value, group_label in [("Control", "Controles"), ("TEA", "TEA")]:
        group = covariates[covariates["group"].eq(group_value)]
        female = group["sexo_femenino"].dropna()
        rows.append(
            {
                "Grupo": group_label,
                "n": int(len(group)),
                "Edad, media (DE)": _mean_sd(group["edad"]),
                "Mujeres, n (%)": (
                    f"{int(female.sum())} ({100 * female.mean():.1f}%)" if len(female) else "NA"
                ),
                "FD medio, media (DE)": _mean_sd(group["movimiento_fd_medio"]),
                "Volumenes FD, %, media (DE)": _mean_sd(group["porcentaje_fd"]),
                "CI total, media (DE)": _mean_sd(group["ci_total"]),
                "CI disponible, n": int(group["ci_total"].notna().sum()),
                "Sitios, n": int(group["sitio"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - float(values.mean())) / std


def build_design_matrix(covariates: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Construye diagnostico + edad + sexo + movimiento + efectos fijos de sitio."""
    required = ["diagnostico_tea", "edad", "sexo_femenino", "movimiento_fd_medio", "sitio"]
    valid = covariates[required].notna().all(axis=1)
    analysis = covariates.loc[valid].reset_index(drop=True).copy()

    site_dummies = pd.get_dummies(analysis["sitio"], prefix="sitio", drop_first=True, dtype=float)
    design = pd.DataFrame(
        {
            "intercepto": 1.0,
            "diagnostico_tea": analysis["diagnostico_tea"].astype(float),
            "edad_z": _standardize(analysis["edad"]),
            "sexo_femenino": analysis["sexo_femenino"].astype(float),
            "movimiento_fd_z": _standardize(analysis["movimiento_fd_medio"]),
        }
    )
    design = pd.concat([design, site_dummies], axis=1)
    return design.to_numpy(dtype=float), analysis, design.columns.tolist()


def fit_mass_univariate_model(
    design: np.ndarray,
    outcomes: np.ndarray,
    coefficient_index: int = 1,
) -> LinearModelResult:
    """Ajusta OLS vectorizado y controla FDR para todos los resultados."""
    design = np.asarray(design, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim == 1:
        outcomes = outcomes[:, None]
    if design.shape[0] != outcomes.shape[0]:
        raise ValueError("Design y outcomes tienen distinto numero de sujetos.")

    xtx_inv = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcomes
    residuals = outcomes - design @ coefficients
    df = int(design.shape[0] - np.linalg.matrix_rank(design))
    if df <= 0:
        raise ValueError("No hay grados de libertad suficientes para el modelo.")

    residual_variance = np.sum(residuals**2, axis=0) / df
    standard_error = np.sqrt(np.maximum(residual_variance * xtx_inv[coefficient_index, coefficient_index], 0))
    beta = coefficients[coefficient_index]
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.divide(beta, standard_error, out=np.zeros_like(beta), where=standard_error > 0)
    p_value = 2.0 * stats.t.sf(np.abs(statistic), df)
    finite = np.isfinite(p_value)
    q_value = np.ones_like(p_value)
    if finite.any():
        _, q_value[finite] = fdrcorrection(p_value[finite], alpha=0.05)
    partial_r = statistic / np.sqrt(statistic**2 + df)
    return LinearModelResult(
        beta=beta,
        standard_error=standard_error,
        statistic=statistic,
        p_value=p_value,
        q_value=q_value,
        partial_r=partial_r,
        degrees_freedom=df,
    )


def _edge_indices(n_rois: int, directed: bool) -> tuple[np.ndarray, np.ndarray]:
    if directed:
        mask = ~np.eye(n_rois, dtype=bool)
        return np.where(mask)
    return np.triu_indices(n_rois, k=1)


def _prepare_edge_values(matrices: np.ndarray, method: str) -> np.ndarray:
    n_rois = matrices.shape[1]
    i_idx, j_idx = _edge_indices(n_rois, method in DIRECTED_METHODS)
    values = np.asarray(matrices[:, i_idx, j_idx], dtype=float)
    if method == "correlacion":
        values = np.arctanh(np.clip(values, -0.999999, 0.999999))
    return values


def _network_ordered_layout(roi_names: Sequence[str]) -> tuple[np.ndarray, list[str], list[float]]:
    network_rank = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    indices = np.arange(len(roi_names))
    networks = np.asarray([_roi_network(name) for name in roi_names], dtype=object)
    order = np.asarray(
        sorted(indices, key=lambda idx: (network_rank.get(str(networks[idx]), 999), idx)),
        dtype=int,
    )
    ordered_networks = networks[order]
    labels: list[str] = []
    centers: list[float] = []
    boundaries: list[float] = []
    start = 0
    while start < len(order):
        network = str(ordered_networks[start])
        end = start + 1
        while end < len(order) and str(ordered_networks[end]) == network:
            end += 1
        labels.append(NETWORK_PUBLIC_LABELS.get(network, network))
        centers.append((start + end - 1) / 2)
        if end < len(order):
            boundaries.append(end - 0.5)
        start = end
    return order, labels, centers + boundaries


def _plot_network_matrix(
    matrix: np.ndarray,
    roi_names: Sequence[str],
    path: Path,
    title: str,
    colorbar_label: str,
    significant: np.ndarray | None = None,
) -> Path:
    order, labels, layout = _network_ordered_layout(roi_names)
    n_labels = len(labels)
    centers = layout[:n_labels]
    boundaries = layout[n_labels:]
    ordered = np.asarray(matrix)[np.ix_(order, order)]
    nonzero = np.abs(ordered[np.nonzero(ordered)])
    vmax = float(np.nanpercentile(nonzero, 99)) if nonzero.size else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=220)
    image = ax.imshow(ordered, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for boundary in boundaries:
        ax.axhline(boundary, color="#323232", linewidth=0.45, alpha=0.7)
        ax.axvline(boundary, color="#323232", linewidth=0.45, alpha=0.7)
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Red de destino")
    ax.set_ylabel("Red de origen")
    ax.set_title(title)

    if significant is not None:
        ordered_sig = np.asarray(significant, dtype=bool)[np.ix_(order, order)]
        y, x = np.where(ordered_sig)
        if len(x):
            ax.scatter(x, y, s=2.5, c="black", marker="s", linewidths=0, alpha=0.8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_edgewise_inference(
    source_dir: str | Path,
    output_dir: str | Path,
    covariates: pd.DataFrame,
    methods: Sequence[str] = PRIMARY_METHODS,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Ajusta comparaciones por conexion para Pearson y Graphical Lasso."""
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    design, analysis_covariates, _ = build_design_matrix(covariates)
    manifest = load_subject_manifest(source_dir)
    valid_ids = analysis_covariates["file_id"].astype(str).tolist()
    manifest_index = {str(file_id): idx for idx, file_id in enumerate(manifest["file_id"].astype(str))}
    selected_idx = np.asarray([manifest_index[file_id] for file_id in valid_ids], dtype=int)

    summaries = []
    paths: Dict[str, str] = {}
    for method in methods:
        data = load_method_data(source_dir, method, manifest)
        matrices = data.matrices[selected_idx]
        values = _prepare_edge_values(matrices, method)
        result = fit_mass_univariate_model(design, values)
        n_rois = matrices.shape[1]
        i_idx, j_idx = _edge_indices(n_rois, data.directed)
        group = analysis_covariates["diagnostico_tea"].to_numpy(dtype=int)

        tea_mean = values[group == 1].mean(axis=0)
        control_mean = values[group == 0].mean(axis=0)
        if method == "correlacion":
            tea_display = np.tanh(tea_mean)
            control_display = np.tanh(control_mean)
        else:
            tea_display = tea_mean
            control_display = control_mean

        table = pd.DataFrame(
            {
                "method": method,
                "roi_i_idx": i_idx,
                "roi_j_idx": j_idx,
                "roi_i": [data.roi_names[i] for i in i_idx],
                "roi_j": [data.roi_names[j] for j in j_idx],
                "network_i": [_roi_network(data.roi_names[i]) for i in i_idx],
                "network_j": [_roi_network(data.roi_names[j]) for j in j_idx],
                "control_mean": control_display,
                "tea_mean": tea_display,
                "adjusted_beta_tea": result.beta,
                "standard_error": result.standard_error,
                "t_statistic": result.statistic,
                "p_value": result.p_value,
                "q_fdr": result.q_value,
                "partial_r": result.partial_r,
                "fdr_significant_005": result.q_value < 0.05,
            }
        ).sort_values(["q_fdr", "p_value", "partial_r"], ascending=[True, True, False])
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        table_path = tables_dir / f"inferencia_ajustada_{method}.csv"
        table.to_csv(table_path, index=False, encoding="utf-8-sig")

        beta_matrix = np.zeros((n_rois, n_rois), dtype=float)
        significance_matrix = np.zeros((n_rois, n_rois), dtype=bool)
        beta_matrix[i_idx, j_idx] = result.beta
        significance_matrix[i_idx, j_idx] = result.q_value < 0.05
        if not data.directed:
            beta_matrix[j_idx, i_idx] = result.beta
            significance_matrix[j_idx, i_idx] = result.q_value < 0.05

        figure_path = figures_dir / f"{method}_efecto_ajustado_fdr.png"
        _plot_network_matrix(
            beta_matrix,
            data.roi_names,
            figure_path,
            f"{PUBLIC_LABELS.get(method, method)}: efecto ajustado TEA - controles",
            "Coeficiente ajustado",
            significant=significance_matrix,
        )
        paths[method] = str(figure_path)
        summaries.append(
            {
                "method": method,
                "public_label": PUBLIC_LABELS.get(method, method),
                "n_subjects": int(len(analysis_covariates)),
                "n_connections": int(values.shape[1]),
                "degrees_freedom": result.degrees_freedom,
                "n_fdr_significant_005": int(np.sum(result.q_value < 0.05)),
                "minimum_p_value": float(np.min(result.p_value)),
                "minimum_q_fdr": float(np.min(result.q_value)),
                "maximum_abs_partial_r": float(np.max(np.abs(result.partial_r))),
                "table": str(table_path),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(tables_dir / "inferencia_ajustada_resumen.csv", index=False, encoding="utf-8-sig")
    return summary, paths


def _component_edges(
    statistic: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    n_rois: int,
    threshold: float,
) -> list[dict[str, object]]:
    selected = np.abs(statistic) >= threshold
    adjacency = np.zeros((n_rois, n_rois), dtype=bool)
    adjacency[i_idx[selected], j_idx[selected]] = True
    adjacency[j_idx[selected], i_idx[selected]] = True
    n_components, labels = connected_components(
        csr_matrix(adjacency.astype(np.int8)),
        directed=False,
        return_labels=True,
    )
    components: list[dict[str, object]] = []
    for component_id in range(n_components):
        nodes = np.where(labels == component_id)[0]
        if len(nodes) < 2:
            continue
        node_mask = np.zeros(n_rois, dtype=bool)
        node_mask[nodes] = True
        edge_mask = selected & node_mask[i_idx] & node_mask[j_idx]
        edge_count = int(edge_mask.sum())
        if edge_count:
            components.append(
                {
                    "nodes": nodes,
                    "edge_mask": edge_mask,
                    "edge_count": edge_count,
                    "max_abs_t": float(np.max(np.abs(statistic[edge_mask]))),
                }
            )
    return sorted(components, key=lambda item: int(item["edge_count"]), reverse=True)


def _residualize(matrix: np.ndarray, covariate_design: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return values - covariate_design @ (np.linalg.pinv(covariate_design) @ values)


def run_nbs_inference(
    source_dir: str | Path,
    output_dir: str | Path,
    covariates: pd.DataFrame,
    methods: Sequence[str] = PRIMARY_METHODS,
    primary_threshold: float = 3.1,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """NBS bilateral con permutacion del diagnostico dentro de cada sitio."""
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    design, analysis_covariates, design_names = build_design_matrix(covariates)
    diagnosis_index = design_names.index("diagnostico_tea")
    covariate_design = np.delete(design, diagnosis_index, axis=1)
    covariate_pinv = np.linalg.pinv(covariate_design)
    group = analysis_covariates["diagnostico_tea"].to_numpy(dtype=float)
    sites = analysis_covariates["sitio"].astype(str).to_numpy()
    group_indices = [np.where(sites == site)[0] for site in np.unique(sites)]
    rng = np.random.default_rng(random_state)

    manifest = load_subject_manifest(source_dir)
    manifest_index = {str(file_id): idx for idx, file_id in enumerate(manifest["file_id"].astype(str))}
    selected_idx = np.asarray(
        [manifest_index[file_id] for file_id in analysis_covariates["file_id"].astype(str)],
        dtype=int,
    )

    summary_rows: list[dict[str, object]] = []
    figure_paths: Dict[str, str] = {}
    for method in methods:
        data = load_method_data(source_dir, method, manifest)
        values = _prepare_edge_values(data.matrices[selected_idx], method)
        n_rois = data.matrices.shape[1]
        i_idx, j_idx = _edge_indices(n_rois, directed=False)
        actual_model = fit_mass_univariate_model(design, values)
        actual_components = _component_edges(
            actual_model.statistic,
            i_idx,
            j_idx,
            n_rois,
            primary_threshold,
        )

        residual_outcomes = values - covariate_design @ (covariate_pinv @ values)
        df = int(len(group) - np.linalg.matrix_rank(covariate_design) - 1)
        null_max_edges = np.zeros(n_permutations, dtype=int)
        for permutation in range(n_permutations):
            permuted_group = group.copy()
            for indices in group_indices:
                permuted_group[indices] = rng.permutation(permuted_group[indices])
            residual_group = (
                permuted_group[:, None]
                - covariate_design @ (covariate_pinv @ permuted_group[:, None])
            ).ravel()
            denominator = float(residual_group @ residual_group)
            if denominator <= 0:
                continue
            beta = residual_group @ residual_outcomes / denominator
            residual = residual_outcomes - np.outer(residual_group, beta)
            variance = np.sum(residual**2, axis=0) / df
            standard_error = np.sqrt(np.maximum(variance / denominator, 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                statistic = np.divide(
                    beta,
                    standard_error,
                    out=np.zeros_like(beta),
                    where=standard_error > 0,
                )
            components = _component_edges(
                statistic,
                i_idx,
                j_idx,
                n_rois,
                primary_threshold,
            )
            null_max_edges[permutation] = int(components[0]["edge_count"]) if components else 0

        nbs_mask = np.zeros(values.shape[1], dtype=bool)
        component_rows: list[dict[str, object]] = []
        edge_rows: list[dict[str, object]] = []
        for component_number, component in enumerate(actual_components, start=1):
            edge_count = int(component["edge_count"])
            p_nbs = float((1 + np.sum(null_max_edges >= edge_count)) / (n_permutations + 1))
            significant = p_nbs < 0.05
            edge_mask = np.asarray(component["edge_mask"], dtype=bool)
            if significant:
                nbs_mask |= edge_mask
            nodes = np.asarray(component["nodes"], dtype=int)
            component_rows.append(
                {
                    "method": method,
                    "public_label": PUBLIC_LABELS.get(method, method),
                    "component": component_number,
                    "n_nodes": int(len(nodes)),
                    "n_edges": edge_count,
                    "max_abs_t": float(component["max_abs_t"]),
                    "p_nbs": p_nbs,
                    "significant_005": significant,
                    "networks": "|".join(
                        sorted({_roi_network(data.roi_names[node]) for node in nodes})
                    ),
                }
            )
            for edge_position in np.where(edge_mask)[0]:
                edge_rows.append(
                    {
                        "method": method,
                        "component": component_number,
                        "p_nbs": p_nbs,
                        "significant_005": significant,
                        "roi_i_idx": int(i_idx[edge_position]),
                        "roi_j_idx": int(j_idx[edge_position]),
                        "roi_i": data.roi_names[int(i_idx[edge_position])],
                        "roi_j": data.roi_names[int(j_idx[edge_position])],
                        "network_i": _roi_network(data.roi_names[int(i_idx[edge_position])]),
                        "network_j": _roi_network(data.roi_names[int(j_idx[edge_position])]),
                        "adjusted_beta_tea": float(actual_model.beta[edge_position]),
                        "t_statistic": float(actual_model.statistic[edge_position]),
                        "p_value": float(actual_model.p_value[edge_position]),
                        "q_fdr": float(actual_model.q_value[edge_position]),
                    }
                )

        components_table = pd.DataFrame(component_rows)
        edges_table = pd.DataFrame(edge_rows)
        components_path = tables_dir / f"nbs_componentes_{method}.csv"
        edges_path = tables_dir / f"nbs_conexiones_{method}.csv"
        components_table.to_csv(components_path, index=False, encoding="utf-8-sig")
        edges_table.to_csv(edges_path, index=False, encoding="utf-8-sig")
        np.save(tables_dir / f"nbs_distribucion_nula_{method}.npy", null_max_edges)

        effect_matrix = np.zeros((n_rois, n_rois), dtype=float)
        effect_matrix[i_idx[nbs_mask], j_idx[nbs_mask]] = actual_model.beta[nbs_mask]
        effect_matrix[j_idx[nbs_mask], i_idx[nbs_mask]] = actual_model.beta[nbs_mask]
        figure_path = figures_dir / f"{method}_componentes_nbs.png"
        _plot_network_matrix(
            effect_matrix,
            data.roi_names,
            figure_path,
            f"{PUBLIC_LABELS.get(method, method)}: componentes NBS significativos",
            "Efecto ajustado TEA - controles",
        )
        figure_paths[method] = str(figure_path)
        significant_components = (
            components_table[components_table["significant_005"]]
            if not components_table.empty
            else components_table
        )
        summary_rows.append(
            {
                "method": method,
                "public_label": PUBLIC_LABELS.get(method, method),
                "primary_threshold_abs_t": primary_threshold,
                "n_permutations": n_permutations,
                "permutation_constraint": "within_site",
                "n_components_observed": int(len(components_table)),
                "n_components_significant_005": int(len(significant_components)),
                "n_edges_significant_components": (
                    int(significant_components["n_edges"].sum())
                    if not significant_components.empty
                    else 0
                ),
                "minimum_p_nbs": (
                    float(components_table["p_nbs"].min())
                    if not components_table.empty
                    else np.nan
                ),
                "components_table": str(components_path),
                "edges_table": str(edges_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "nbs_resumen.csv", index=False, encoding="utf-8-sig")
    return summary, figure_paths


def build_subject_level_metrics(
    source_dir: str | Path,
    output_dir: str | Path,
    covariates: pd.DataFrame,
    methods: Sequence[str] = ALL_METHODS,
    threshold: float = SPARSITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Calcula magnitud y esparsidad en cada sujeto y ajusta diferencias grupales."""
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_subject_manifest(source_dir)

    rows = []
    for method in methods:
        data = load_method_data(source_dir, method, manifest)
        values = _prepare_edge_values(data.matrices, method)
        if method == "correlacion":
            display_values = np.tanh(values)
        else:
            display_values = values
        for idx, row in data.manifest.iterrows():
            subject_values = display_values[idx]
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "file_id": row["file_id"],
                    "group": row["group"],
                    "grupo": "TEA" if row["group"] == "TEA" else "Controles",
                    "site": row["site"],
                    "method": method,
                    "public_label": PUBLIC_LABELS.get(method, method),
                    "mean_abs_connectivity": float(np.mean(np.abs(subject_values))),
                    "sparsity_001": float(np.mean(np.abs(subject_values) <= threshold)),
                    "n_active_connections": int(np.sum(np.abs(subject_values) > threshold)),
                    "n_connections": int(subject_values.size),
                }
            )
    metrics = pd.DataFrame(rows)
    metrics_path = tables_dir / "metricas_conectividad_por_sujeto.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    design, analysis_covariates, _ = build_design_matrix(covariates)
    valid_order = analysis_covariates["file_id"].astype(str).tolist()
    summary_rows = []
    for method in methods:
        method_df = metrics[metrics["method"].eq(method)].set_index("file_id").loc[valid_order]
        for metric in ["mean_abs_connectivity", "sparsity_001"]:
            values = method_df[metric].to_numpy(dtype=float)
            result = fit_mass_univariate_model(design, values)
            control = values[analysis_covariates["diagnostico_tea"].to_numpy() == 0]
            tea = values[analysis_covariates["diagnostico_tea"].to_numpy() == 1]
            summary_rows.append(
                {
                    "method": method,
                    "public_label": PUBLIC_LABELS.get(method, method),
                    "metric": metric,
                    "control_mean": float(control.mean()),
                    "control_sd": float(control.std(ddof=1)),
                    "tea_mean": float(tea.mean()),
                    "tea_sd": float(tea.std(ddof=1)),
                    "adjusted_beta_tea": float(result.beta[0]),
                    "standard_error": float(result.standard_error[0]),
                    "p_value": float(result.p_value[0]),
                    "partial_r": float(result.partial_r[0]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    _, summary["q_fdr"] = fdrcorrection(summary["p_value"].to_numpy(), alpha=0.05)
    summary_path = tables_dir / "metricas_conectividad_por_sujeto_resumen.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_methods = [method for method in methods if method in PRIMARY_METHODS]
    figure_path = figures_dir / "distribuciones_conectividad_por_sujeto.png"
    fig, axes = plt.subplots(
        2,
        len(plot_methods),
        figsize=(9.5, 7.4),
        dpi=220,
        squeeze=False,
    )
    colors = {"Controles": "#4c78a8", "TEA": "#e45756"}
    rng = np.random.default_rng(42)
    for col, method in enumerate(plot_methods):
        method_df = metrics[metrics["method"].eq(method)]
        for row_idx, (metric, ylabel) in enumerate(
            [
                ("mean_abs_connectivity", "Magnitud absoluta media"),
                ("sparsity_001", "Esparsidad (|peso| <= 0,001)"),
            ]
        ):
            ax = axes[row_idx, col]
            values = [
                method_df.loc[method_df["grupo"].eq(group), metric].to_numpy(dtype=float)
                for group in ["Controles", "TEA"]
            ]
            violin = ax.violinplot(values, positions=[1, 2], showmeans=False, showmedians=True)
            for body, group in zip(violin["bodies"], ["Controles", "TEA"]):
                body.set_facecolor(colors[group])
                body.set_edgecolor("#333333")
                body.set_alpha(0.55)
            for pos, group_values, group in zip([1, 2], values, ["Controles", "TEA"]):
                sample_size = min(120, len(group_values))
                sample = rng.choice(group_values, size=sample_size, replace=False)
                jitter = rng.normal(pos, 0.035, size=sample_size)
                ax.scatter(jitter, sample, s=5, alpha=0.25, color=colors[group], linewidths=0)
            ax.set_xticks([1, 2], ["Controles", "TEA"])
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(PUBLIC_LABELS.get(method, method))
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Distribuciones de conectividad calculadas por sujeto", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return metrics, summary, str(figure_path)


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def build_classification_intervals_and_roc(
    source_dir: str | Path,
    output_dir: str | Path,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Calcula IC95% por bootstrap de sitios y regenera la curva ROC en espanol."""
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    predictions = pd.read_csv(source_dir / "predicciones_por_sujeto.csv")
    predictions["method_label"] = predictions["method"].map(
        {
            "pearson": "Pearson",
            "graphical_lasso": "Graphical Lasso",
            "lingam": "LiNGAM",
        }
    )
    predictions["classifier_label"] = predictions["classifier"].map(
        {"svm": "SVM", "rf": "Random Forest"}
    )
    rng = np.random.default_rng(random_state)
    rows = []

    fig, ax = plt.subplots(figsize=(8.2, 6.3), dpi=220)
    for (method, classifier), group in predictions.groupby(["method", "classifier"], sort=False):
        group = group.reset_index(drop=True)
        y_true = group["true_label"].to_numpy(dtype=int)
        y_score = group["prob_asd"].to_numpy(dtype=float)
        point = _classification_metrics(y_true, y_score)
        sites = group["site"].astype(str).unique()
        bootstrap = {metric: [] for metric in point}

        for _ in range(n_bootstrap):
            sampled_sites = rng.choice(sites, size=len(sites), replace=True)
            sampled_parts = [group[group["site"].astype(str).eq(site)] for site in sampled_sites]
            sampled = pd.concat(sampled_parts, ignore_index=True)
            sampled_y = sampled["true_label"].to_numpy(dtype=int)
            if np.unique(sampled_y).size < 2:
                continue
            sampled_metrics = _classification_metrics(
                sampled_y,
                sampled["prob_asd"].to_numpy(dtype=float),
            )
            for metric, value in sampled_metrics.items():
                bootstrap[metric].append(value)

        method_label = str(group["method_label"].iloc[0])
        classifier_label = str(group["classifier_label"].iloc[0])
        for metric, point_value in point.items():
            values = np.asarray(bootstrap[metric], dtype=float)
            rows.append(
                {
                    "method": method,
                    "public_label": method_label,
                    "classifier": classifier,
                    "classifier_label": classifier_label,
                    "metric": metric,
                    "estimate": float(point_value),
                    "ci95_lower": float(np.percentile(values, 2.5)),
                    "ci95_upper": float(np.percentile(values, 97.5)),
                    "bootstrap_unit": "site",
                    "n_bootstrap_valid": int(len(values)),
                }
            )

        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(
            fpr,
            tpr,
            linewidth=1.8,
            label=f"{method_label} + {classifier_label} (AUC={point['auc']:.2f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC para la clasificación de TEA y controles")
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    roc_path = output_dir / "figures" / "curvas_roc_tea_controles.png"
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)

    intervals = pd.DataFrame(rows)
    intervals_path = output_dir / "tables" / "metricas_clasificacion_intervalos.csv"
    intervals_path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(intervals_path, index=False, encoding="utf-8-sig")
    return intervals, str(roc_path)


def run_statistical_analysis(
    source_dir: str | Path,
    output_dir: str | Path,
    phenotype_csv: str | Path,
) -> Dict[str, object]:
    """Ejecuta todas las salidas estadisticas derivadas."""
    source_dir = _as_path(source_dir)
    output_dir = _as_path(output_dir)
    covariates = load_covariates(source_dir, phenotype_csv)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    covariate_path = tables_dir / "covariables_por_sujeto.csv"
    covariates.to_csv(covariate_path, index=False, encoding="utf-8-sig")
    covariate_summary = build_covariate_summary(covariates)
    covariate_summary_path = tables_dir / "resumen_covariables_grupo.csv"
    covariate_summary.to_csv(covariate_summary_path, index=False, encoding="utf-8-sig")

    inference_summary, inference_figures = run_edgewise_inference(
        source_dir,
        output_dir,
        covariates,
    )
    nbs_summary, nbs_figures = run_nbs_inference(
        source_dir,
        output_dir,
        covariates,
    )
    _, subject_summary, subject_figure = build_subject_level_metrics(
        source_dir,
        output_dir,
        covariates,
    )
    classification_intervals, roc_path = build_classification_intervals_and_roc(
        source_dir,
        output_dir,
    )
    return {
        "covariates": str(covariate_path),
        "covariate_summary": str(covariate_summary_path),
        "inference_summary": inference_summary,
        "inference_figures": inference_figures,
        "nbs_summary": nbs_summary,
        "nbs_figures": nbs_figures,
        "subject_metric_summary": subject_summary,
        "subject_metric_figure": subject_figure,
        "classification_intervals": classification_intervals,
        "roc_figure": roc_path,
    }
