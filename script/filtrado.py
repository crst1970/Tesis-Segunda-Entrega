"""
filtrado.py
-----------
Filtro pasa-banda y z-score para señales fMRI en reposo.
Frecuencias estándar: 0.01–0.1 Hz (banda de interés en resting-state).

Funciones:
  - get_tr            : extrae el TR real desde el header NIfTI del sujeto
  - get_tr_ms         : igual pero para headers en milisegundos
  - bandpass_filter   : filtro pasa-banda 1D (un voxel o ROI)
  - filtrar_rois      : aplica bandpass a toda la matriz (T x n_rois)
  - zscore_rois       : estandariza cada ROI (media=0, std=1)
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ─────────────────────────────────────────────────────────────────────────────
# TR por sujeto
# ─────────────────────────────────────────────────────────────────────────────

def get_tr(fmri_img):
    """
    Extrae el Repetition Time (TR) desde el header NIfTI de la imagen fMRI.

    El TR varía según el sitio de adquisición en datasets multi-sitio como
    ABIDE (ej: NYU usa 2.0s, UCLA 3.0s, UM 2.0s, etc.). Por eso nunca debe
    asumirse un TR fijo — siempre extraerlo del header de cada sujeto.

    El header NIfTI guarda el TR en la 4ta entrada de get_zooms(), que
    corresponde al tamaño del voxel en la dimensión temporal.

    Parámetros
    ----------
    fmri_img : Nifti1Image — imagen cargada con nibabel

    Retorna
    -------
    tr : float — Repetition Time en segundos
    """
    zooms = fmri_img.header.get_zooms()

    if len(zooms) < 4:
        raise ValueError(
            f'El header NIfTI tiene {len(zooms)} dimensiones, se esperan 4 (X, Y, Z, T). '
            'Verificá que la imagen sea 4D.'
        )

    tr = float(zooms[3])

    if tr <= 0:
        raise ValueError(
            f'TR inválido: {tr}s. El header puede estar corrupto o en unidades incorrectas. '
            'Algunos datasets guardan el TR en milisegundos — si obtenés valores como 2000, '
            'usá get_tr_ms().'
        )

    # Advertencia si el TR parece estar en milisegundos
    if tr > 20:
        print(f'[ADVERTENCIA] TR={tr}s parece muy alto. Si está en milisegundos, usá get_tr_ms().')

    print(f'TR extraído del header: {tr}s')
    return tr


def get_tr_ms(fmri_img):
    """
    Igual que get_tr() pero para imágenes cuyo header guarda el TR
    en milisegundos (algunos datasets no siguen el estándar NIfTI).

    Retorna
    -------
    tr : float — Repetition Time en segundos (convertido desde ms)
    """
    tr_ms = float(fmri_img.header.get_zooms()[3])
    tr_s  = tr_ms / 1000.0
    print(f'TR extraído del header: {tr_ms}ms → {tr_s}s')
    return tr_s


# ─────────────────────────────────────────────────────────────────────────────
# Filtrado
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut=0.01, highcut=0.1, tr=2.0, order=4):
    """
    Filtro pasa-banda Butterworth para una señal temporal 1D.

    Retiene las frecuencias entre lowcut y highcut Hz, eliminando:
      - Drift lento del escáner (< 0.01 Hz)
      - Ruido fisiológico: respiración (~0.3 Hz) y latido (~1 Hz)

    Usa filtfilt (filtrado de fase cero): aplica el filtro dos veces,
    ida y vuelta, para evitar desfase temporal en la señal resultante.

    Parámetros
    ----------
    signal  : array (T,)  — señal temporal de un voxel o ROI
    lowcut  : float       — frecuencia de corte inferior en Hz (default 0.01)
    highcut : float       — frecuencia de corte superior en Hz (default 0.1)
    tr      : float       — TR del sujeto en segundos (usar get_tr() por sujeto)
    order   : int         — orden del filtro Butterworth (default 4)

    Retorna
    -------
    array (T,) — señal filtrada
    """
    fs  = 1.0 / tr
    nyq = fs / 2.0

    low  = max(lowcut  / nyq, 1e-6)
    high = min(highcut / nyq, 0.9999)

    if low >= high:
        raise ValueError(
            f'Rango inválido: lowcut={lowcut}Hz >= highcut={highcut}Hz '
            f'con TR={tr}s (Nyquist={nyq:.4f}Hz). Reducí highcut o aumentá lowcut.'
        )

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def filtrar_rois(roi_signals, tr, lowcut=0.01, highcut=0.1, order=4):
    """
    Aplica bandpass_filter a todas las ROIs de la matriz (T x n_rois).

    Parámetros
    ----------
    roi_signals : array (T, n_rois) — señales crudas por ROI
    tr          : float             — TR del sujeto (usar get_tr() por sujeto)
    lowcut      : float             — corte inferior en Hz (default 0.01)
    highcut     : float             — corte superior en Hz (default 0.1)
    order       : int               — orden del filtro (default 4)

    Retorna
    -------
    array (T, n_rois) — señales filtradas
    """
    return np.array([
        bandpass_filter(roi_signals[:, i], lowcut=lowcut, highcut=highcut,
                        tr=tr, order=order)
        for i in range(roi_signals.shape[1])
    ]).T


# ─────────────────────────────────────────────────────────────────────────────
# Z-score
# ─────────────────────────────────────────────────────────────────────────────

def zscore_rois(roi_signals):
    """
    Estandariza cada ROI a media=0 y std=1 (z-score por columna).

    Aplicar z-score antes de calcular conectividad es estándar en muchos
    pipelines: pone todas las ROIs en la misma escala, evitando que ROIs
    con mayor amplitud dominen la matriz de conectividad.

    Parámetros
    ----------
    roi_signals : array (T, n_rois) — señales (filtradas o crudas)

    Retorna
    -------
    array (T, n_rois) — señales estandarizadas

    Notas
    -----
    ROIs con std=0 se dejan en cero en lugar de generar NaN por división
    por cero (ocurre en señales constantes, típico fuera del cerebro).
    """
    mean     = roi_signals.mean(axis=0)
    std      = roi_signals.std(axis=0)
    std_safe = np.where(std == 0, 1, std)

    zscored = (roi_signals - mean) / std_safe

    n_cero = np.sum(std == 0)
    if n_cero > 0:
        print(f'[AVISO] {n_cero} ROI(s) con std=0 dejadas en cero.')

    return zscored