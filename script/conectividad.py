"""
conectividad.py
---------------
Métodos de conectividad funcional sobre señales ROI para fMRI resting-state.

MÉTODOS ASOCIATIVOS (simétricos — matrix[i,j] == matrix[j,i]):
  1. correlacion          : Correlación de Pearson
  2. correlacion_parcial  : Correlación parcial (Nilearn)
  3. graphical_lasso      : GL real con penalización L1 (sklearn GraphicalLassoCV)

MÉTODOS CAUSALES (asimétricos — matrix[i,j] ≠ matrix[j,i]):
  4. granger              : Granger causality (statsmodels)
  5. pcmci                : PCMCI — causalidad para series temporales (tigramite)

UTILIDADES:
  - umbralizar            : pone en cero entradas bajo el umbral
  - vectorizar            : extrae features de la matriz para el clasificador
  - comparar_matrices     : resumen estadístico comparativo

Dependencias:
  pip install nilearn scikit-learn statsmodels tigramite
"""

import warnings

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import grangercausalitytests


# ─────────────────────────────────────────────────────────────────────────────
# MÉTODOS ASOCIATIVOS
# ─────────────────────────────────────────────────────────────────────────────

def correlacion(roi_signals_filt):
    """
    Correlación de Pearson entre ROIs.

    Mide cuán similares son las señales temporales de cada par de ROIs.
    Es el método más simple y el baseline universal en conectividad fMRI.

    Limitación: no distingue conexiones directas de indirectas. Si A y B
    están ambas conectadas a C, aparecerán correlacionadas entre sí aunque
    no tengan conexión directa.

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois) — señales filtradas y z-score

    Retorna
    -------
    corr_matrix : array (n_rois, n_rois) — valores en [-1, 1], simétrica
    """
    return np.corrcoef(roi_signals_filt.T)


def correlacion_parcial(roi_signals_filt):
    """
    Correlación parcial entre ROIs.

    Correlación entre ROI_i y ROI_j controlando el efecto lineal de todas
    las demás. Reduce conexiones indirectas sin penalización sparse.
    Punto intermedio entre Pearson (no controla nada) y GL (muy sparse).

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois)

    Retorna
    -------
    pc_matrix : array (n_rois, n_rois) — valores en [-1, 1], simétrica
    """
    measure = ConnectivityMeasure(kind='partial correlation')
    return measure.fit_transform([roi_signals_filt])[0]


def _graphical_lasso_legacy(roi_signals_filt, cv=5, max_iter=500):
    """
    Graphical Lasso real con penalización L1 (sklearn GraphicalLassoCV).

    IMPORTANTE: kind='precision' de Nilearn NO es Graphical Lasso real —
    solo invierte la covarianza muestral sin penalización. Esta función usa
    GraphicalLassoCV de sklearn, que aplica L1 y fuerza entradas a exactamente
    cero (sparse), separando conexiones directas de indirectas con precisión.

    La precisión se normaliza para que sea comparable con Pearson:
        gl_norm[i,j] = -prec[i,j] / sqrt(prec[i,i] * prec[j,j])
    con diagonal forzada a 1.0  →  valores en [-1, 1].

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois)
    cv               : int   — folds para elegir alpha óptimo (default 5)
    max_iter         : int   — iteraciones máximas (default 500)

    Retorna
    -------
    gl_norm : array (n_rois, n_rois) — valores en [-1, 1], simétrica, sparse
    alpha_  : float — alpha de regularización L1 elegido por CV
    """
    model = GraphicalLassoCV(cv=cv, max_iter=max_iter)
    model.fit(roi_signals_filt)
    prec = model.precision_

    d       = np.sqrt(np.diag(prec))
    gl_norm = -prec / np.outer(d, d)
    np.fill_diagonal(gl_norm, 1.0)

    n_rois = roi_signals_filt.shape[1]
    upper  = np.triu_indices(n_rois, k=1)
    n_nz   = np.sum(gl_norm[upper] != 0)
    n_tot  = len(upper[0])
    print(f'GraphicalLasso: alpha={model.alpha_:.4f} | '
          f'{n_nz}/{n_tot} entradas no nulas ({100*n_nz/n_tot:.1f}% sparse)')

    return gl_norm, model.alpha_


def graphical_lasso(roi_signals_filt, cv=None, max_iter=500, alpha=0.2):
    """
    Graphical Lasso robusto para fMRI.

    Si hay ROIs con NaN/inf o varianza cero/casi cero, se excluyen del ajuste y
    quedan en cero en la matriz final. Por defecto usa alpha fijo para evitar
    el costo de GraphicalLassoCV sujeto por sujeto. Si cv no es None, intenta CV.
    """
    x = np.asarray(roi_signals_filt, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"GraphicalLasso espera array 2D (T, n_rois); recibido shape={x.shape}")

    T, n_rois = x.shape
    finite_roi = np.all(np.isfinite(x), axis=0)
    variances = np.nanvar(x, axis=0)
    valid_roi = finite_roi & (variances > 1e-12)
    valid_idx = np.where(valid_roi)[0]
    n_valid = len(valid_idx)
    n_bad = n_rois - n_valid

    if n_bad:
        print(
            f"[AVISO] GraphicalLasso: {n_bad} ROI(s) con NaN/inf o varianza cero/casi cero. "
            "Se excluyen del ajuste y quedan en 0."
        )
    if n_valid < 2:
        raise ValueError("GraphicalLasso no tiene suficientes ROIs validas despues de filtrar varianza.")

    x_valid = x[:, valid_roi]
    x_valid = x_valid - np.mean(x_valid, axis=0, keepdims=True)
    std = np.std(x_valid, axis=0, ddof=0, keepdims=True)
    std[std < 1e-12] = 1.0
    x_valid = x_valid / std

    alpha_used = np.nan
    fit_mode = "GraphicalLasso_fixed"

    try:
        if cv is None:
            model = GraphicalLasso(
                alpha=alpha,
                max_iter=max_iter,
                assume_centered=True,
            )
        else:
            cv_eff = max(2, min(int(cv), T - 1))
            fit_mode = "GraphicalLassoCV"
            model = GraphicalLassoCV(cv=cv_eff, max_iter=max_iter, assume_centered=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_valid)
        prec_valid = np.asarray(model.precision_, dtype=float)
        alpha_used = float(model.alpha_ if hasattr(model, "alpha_") else alpha)
        if not np.isfinite(prec_valid).all():
            raise FloatingPointError("GraphicalLasso produjo precision no finita.")
    except Exception as exc:
        fit_mode = "GraphicalLasso_fallback"
        fallback_errors = []
        prec_valid = None
        for alpha in (0.05, 0.1, 0.2, 0.5, 1.0):
            try:
                model = GraphicalLasso(
                    alpha=alpha,
                    max_iter=max_iter,
                    assume_centered=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model.fit(x_valid)
                candidate = np.asarray(model.precision_, dtype=float)
                if np.isfinite(candidate).all():
                    prec_valid = candidate
                    alpha_used = float(alpha)
                    break
                fallback_errors.append(f"alpha={alpha}: precision no finita")
            except Exception as fallback_exc:
                fallback_errors.append(f"alpha={alpha}: {fallback_exc}")
        if prec_valid is None:
            details = "; ".join(fallback_errors[:3])
            raise RuntimeError(
                "GraphicalLasso fallo incluso con fallback regularizado. "
                f"Error original: {exc}. Fallbacks: {details}"
            ) from exc
        print(
            "[AVISO] GraphicalLasso fue inestable para este sujeto; "
            f"se uso fallback regularizado con alpha={alpha_used:.4f}."
        )

    diag = np.diag(prec_valid)
    diag = np.where(diag > 1e-12, diag, np.nan)
    denom = np.sqrt(np.outer(diag, diag))
    gl_valid = -prec_valid / denom
    gl_valid = np.nan_to_num(gl_valid, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(gl_valid, 1.0)

    gl_norm = np.zeros((n_rois, n_rois), dtype=float)
    gl_norm[np.ix_(valid_idx, valid_idx)] = gl_valid
    np.fill_diagonal(gl_norm, 1.0)
    gl_norm = np.clip(gl_norm, -1.0, 1.0)

    upper = np.triu_indices(n_rois, k=1)
    n_nz = int(np.sum(np.abs(gl_norm[upper]) > 1e-12))
    n_tot = len(upper[0])
    print(
        f"GraphicalLasso ({fit_mode}): alpha={alpha_used:.4f} | "
        f"{n_nz}/{n_tot} entradas no nulas ({100*n_nz/n_tot:.1f}%)"
    )

    return gl_norm, alpha_used


# ─────────────────────────────────────────────────────────────────────────────
# MÉTODOS CAUSALES
# ─────────────────────────────────────────────────────────────────────────────

def granger(roi_signals_filt, maxlag=2, significance=0.05):
    """
    Granger causality entre todas las parejas de ROIs.

    Pregunta: la historia pasada de ROI_i mejora la predicción de ROI_j
    más allá de lo que predice la propia historia de ROI_j?

    Resultado ASIMÉTRICO: matrix[i,j] = influencia de i sobre j.
    Eso captura la dirección de la influencia temporal — diferencia clave
    frente a todos los métodos asociativos (que son simétricos).

    Valores: -log10(p-value). Umbral de significancia: -log10(0.05) ≈ 1.3.
    Limitación: asume linealidad y estacionariedad. No controla
    autocorrelación de las señales (PCMCI lo hace mejor).

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois)
    maxlag           : int   — lags a evaluar en TRs (default 2)
    significance     : float — umbral p-value para reporte (default 0.05)

    Retorna
    -------
    granger_matrix : array (n_rois, n_rois) — -log10(p-value), asimétrica
                     granger_matrix[i, j] = intensidad de influencia i → j
    pval_matrix    : array (n_rois, n_rois) — p-values crudos
    """
    T, n_rois = roi_signals_filt.shape
    pval_matrix = np.ones((n_rois, n_rois))

    print(f'Calculando Granger ({n_rois}x{n_rois} pares, maxlag={maxlag})...')

    for i in range(n_rois):
        for j in range(n_rois):
            if i == j:
                continue
            # statsmodels espera columnas [efecto, causa] = [ROI_j, ROI_i]
            data = np.column_stack([roi_signals_filt[:, j],
                                    roi_signals_filt[:, i]])
            try:
                res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                pval_matrix[i, j] = res[1][0]['ssr_ftest'][1]
            except Exception:
                pval_matrix[i, j] = 1.0

    granger_matrix = -np.log10(np.clip(pval_matrix, 1e-10, 1.0))
    np.fill_diagonal(granger_matrix, 0)

    mask_off = ~np.eye(n_rois, dtype=bool)
    n_sig    = np.sum(pval_matrix[mask_off] < significance)
    n_total  = n_rois * (n_rois - 1)
    print(f'Granger listo. Conexiones p<{significance}: '
          f'{n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)')
    print(f'Asimetrica: {not np.allclose(granger_matrix, granger_matrix.T)}')

    return granger_matrix, pval_matrix


def pcmci(roi_signals_filt, tau_max=2, pc_alpha=0.05):
    """
    PCMCI (Peter-Clark + Momentary Conditional Independence).

    Diseñado específicamente para series temporales causales. Ventajas
    sobre Granger para fMRI:
      - Controla autocorrelación de las señales (problema frecuente en fMRI)
      - Detecta conexiones a múltiples escalas temporales (tau=1,2,...)
      - Reduce falsos positivos en datos con alta correlación cruzada
      - Produce p-values corregidos por múltiples comparaciones

    La matriz resultante es ASIMÉTRICA: pcmci_matrix[i,j] representa la
    influencia causal de ROI_i sobre ROI_j (tomando el lag más significativo).

    Requiere: pip install tigramite

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois)
    tau_max          : int   — lag máximo en TRs a evaluar (default 2)
    pc_alpha         : float — nivel de significancia (default 0.05)

    Retorna
    -------
    pcmci_matrix : array (n_rois, n_rois) — -log10(p_min sobre lags), asimétrica
    val_matrix   : array (n_rois, n_rois, tau_max+1) — coeficientes por lag
    pval_matrix  : array (n_rois, n_rois, tau_max+1) — p-values por lag
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        raise ImportError(
            'tigramite no esta instalado. '
            'Instalalo con: pip install tigramite'
        )

    n_rois = roi_signals_filt.shape[1]
    print(f'Calculando PCMCI ({n_rois} ROIs, tau_max={tau_max})...')

    df    = pp.DataFrame(roi_signals_filt)
    model = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res   = model.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    pval_matrix = res['p_matrix']   # (n_rois, n_rois, tau_max+1)
    val_matrix  = res['val_matrix']

    # Tomar el p-value minimo entre lags tau=1..tau_max (excluir tau=0)
    pmin         = pval_matrix[:, :, 1:].min(axis=2)
    pcmci_matrix = -np.log10(np.clip(pmin, 1e-10, 1.0))
    np.fill_diagonal(pcmci_matrix, 0)

    mask_off = ~np.eye(n_rois, dtype=bool)
    n_sig    = np.sum(pmin[mask_off] < pc_alpha)
    n_total  = n_rois * (n_rois - 1)
    print(f'PCMCI listo. Conexiones significativas: '
          f'{n_sig}/{n_total} ({100*n_sig/n_total:.1f}%)')
    print(f'Asimetrica: {not np.allclose(pcmci_matrix, pcmci_matrix.T)}')

    return pcmci_matrix, val_matrix, pval_matrix


def lingam(roi_signals_filt, max_iter=1000, significance=0.05, random_state=42):
    """
    DirectLiNGAM para conectividad causal contemporanea entre ROIs.

    Devuelve una matriz asimetrica con la convencion:
    matrix[i, j] = influencia estimada de ROI_i sobre ROI_j.
    """
    try:
        import lingam as lg
    except ImportError as exc:
        raise ImportError(
            "lingam no esta instalado. Instalalo con: pip install lingam"
        ) from exc

    x = np.asarray(roi_signals_filt, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"LiNGAM espera un array 2D (T, n_rois); recibido shape={x.shape}")

    T, n_rois = x.shape
    if n_rois < 2:
        raise ValueError("LiNGAM requiere al menos 2 ROIs.")

    finite_roi = np.all(np.isfinite(x), axis=0)
    variances = np.nanvar(x, axis=0)
    valid_roi = finite_roi & (variances > 1e-12)
    valid_idx = np.where(valid_roi)[0]
    n_valid = len(valid_idx)
    n_bad = n_rois - n_valid

    if n_bad:
        print(
            f"[AVISO] LiNGAM: {n_bad} ROI(s) con NaN/inf o varianza cero/casi cero; "
            "se excluyen del ajuste y quedan en cero en la matriz final."
        )
    if n_valid < 2:
        raise ValueError("LiNGAM no tiene suficientes ROIs validas despues de filtrar varianza.")
    if T <= n_valid:
        raise ValueError(
            f"LiNGAM requiere T > ROIs validas para esta configuracion: "
            f"T={T}, ROIs validas={n_valid}. Usa min_timepoints mayor, max_rois menor "
            "o trata LiNGAM como analisis exploratorio."
        )

    x_valid = x[:, valid_roi]
    print(f"Calculando DirectLiNGAM ({n_valid}/{n_rois} ROIs validas, T={T}, max_iter={max_iter})...")

    try:
        model = lg.DirectLiNGAM(random_state=random_state, max_iter=max_iter)
    except TypeError:
        model = lg.DirectLiNGAM(random_state=random_state)
    model.fit(x_valid)

    B_valid = np.asarray(model.adjacency_matrix_, dtype=float).T
    np.fill_diagonal(B_valid, 0)

    B_matrix = np.zeros((n_rois, n_rois), dtype=float)
    B_matrix[np.ix_(valid_idx, valid_idx)] = B_valid
    np.fill_diagonal(B_matrix, 0)

    max_val = float(np.max(np.abs(B_matrix)))
    if max_val > 0:
        lingam_matrix = B_matrix / max_val
    else:
        lingam_matrix = B_matrix.copy()
    np.fill_diagonal(lingam_matrix, 0)

    raw_order = list(getattr(model, "causal_order_", []))
    causal_order = [int(valid_idx[i]) for i in raw_order] if raw_order else []
    mask_off = ~np.eye(n_rois, dtype=bool)
    n_nz = int(np.sum(np.abs(lingam_matrix[mask_off]) > significance))
    n_total = n_rois * (n_rois - 1)

    print("DirectLiNGAM listo.")
    print(f"  Orden causal (primeras 5 ROIs): {causal_order[:5]}...")
    print(f"  Conexiones |coef|>{significance}: {n_nz}/{n_total} ({100*n_nz/n_total:.1f}%)")
    print(f"  Asimetrica: {not np.allclose(lingam_matrix, lingam_matrix.T)}")

    return lingam_matrix, B_matrix, causal_order


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def umbralizar(matrix, threshold, absoluto=True):
    """
    Pone en cero las entradas que no superan el umbral.

    Parámetros
    ----------
    matrix    : array (n_rois, n_rois)
    threshold : float
    absoluto  : bool — si True, umbral sobre |matrix| (default True)

    Retorna
    -------
    array (n_rois, n_rois) — no modifica la original
    """
    result = matrix.copy()
    result[(np.abs(result) if absoluto else result) < threshold] = 0

    mask_off = ~np.eye(matrix.shape[0], dtype=bool)
    n_nz     = np.sum(result[mask_off] != 0)
    n_tot    = mask_off.sum()
    print(f'Umbral={threshold}: {n_nz}/{n_tot} entradas no nulas '
          f'({100*n_nz/n_tot:.1f}%)')
    return result


def vectorizar(matrix, simetrica=True):
    """
    Convierte la matriz de conectividad en vector de features para el clasificador.

    Matrices simetricas  (Pearson, corr. parcial, GL):
        triángulo superior sin diagonal  →  n*(n-1)/2 features
        Con 148 ROIs: 148*147/2 = 10.878 features

    Matrices asimetricas (Granger, PCMCI):
        toda la matriz sin diagonal      →  n*(n-1) features
        Con 148 ROIs: 148*147 = 21.756 features (el doble, porque i→j ≠ j→i)

    Parámetros
    ----------
    matrix    : array (n_rois, n_rois)
    simetrica : bool — True para asociativos, False para causales

    Retorna
    -------
    vector : array (n_features,)
    """
    n = matrix.shape[0]
    if simetrica:
        return matrix[np.triu_indices(n, k=1)]
    return matrix[~np.eye(n, dtype=bool)]


def comparar_matrices(matrices_dict, n_rois):
    """
    Resumen estadístico comparativo de todas las matrices.

    Parámetros
    ----------
    matrices_dict : dict {nombre: array (n_rois, n_rois)}
    n_rois        : int
    """
    upper    = np.triu_indices(n_rois, k=1)
    mask_off = ~np.eye(n_rois, dtype=bool)

    for nombre, matrix in matrices_dict.items():
        simetrica = np.allclose(matrix, matrix.T, atol=1e-6)
        vals      = matrix[upper] if simetrica else matrix[mask_off]
        n_nz      = np.sum(vals != 0)

        print(f'\n=== {nombre} ===')
        print(f'  Tipo      : {"Simetrica (asociativo)" if simetrica else "Asimetrica (causal)"}')
        print(f'  Features  : {len(vals)}')
        print(f'  Media     : {vals.mean():.4f}')
        print(f'  Std       : {vals.std():.4f}')
        print(f'  Rango     : [{vals.min():.4f}, {vals.max():.4f}]')
        print(f'  No nulas  : {n_nz}/{len(vals)} ({100*n_nz/len(vals):.1f}%)')
