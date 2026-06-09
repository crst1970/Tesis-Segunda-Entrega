"""
parcelacion.py
--------------
Carga atlas, remuestrea al espacio del fMRI, y extrae señal promedio por ROI.

Funciones:
  - cargar_atlas           : descarga y remuestrea el atlas al espacio del sujeto
  - extraer_senales_roi    : extrae señal promedio por ROI
  - precalcular_cache_roi  : pre-calcula señal orig+filtrada para los visores

Atlas disponibles (pasar como atlas_name):
  - 'cort-maxprob-thr25-2mm'   Harvard-Oxford cortical  (default, ~48 regiones)
  - 'sub-maxprob-thr25-2mm'    Harvard-Oxford subcortical
  - 'AAL'                      Automated Anatomical Labeling (~116 regiones)
  - 'destrieux_2009'           Destrieux (~148 regiones)
"""

import numpy as np
from nilearn import datasets
from nilearn.image import resample_to_img


ATLAS_DEFAULT = "schaefer_100"
ATLAS_DISPONIBLES = (
    "schaefer_100",
    "cort-maxprob-thr25-2mm",
    "sub-maxprob-thr25-2mm",
    "AAL",
    "destrieux_2009",
)


def nombre_roi_atlas(atlas_obj, rid):
    """Devuelve el nombre de una ROI respetando si el atlas incluye fondo."""
    labels = [label.decode("utf-8") if isinstance(label, bytes) else str(label) for label in atlas_obj.labels]
    if labels and labels[0].strip().lower() in {"background", "fondo"}:
        idx = int(rid)
    else:
        idx = int(rid) - 1
    return labels[idx] if 0 <= idx < len(labels) else f"ROI_{rid}"


# ─────────────────────────────────────────────────────────────────────────────
# Carga de atlas
# ─────────────────────────────────────────────────────────────────────────────

def cargar_atlas(fmri_img, atlas_name=ATLAS_DEFAULT):
    """
    Descarga un atlas y lo remuestrea al espacio del fMRI del sujeto.

    El atlas vive en espacio MNI estándar (2mm). El fMRI de cada sujeto
    puede tener diferente resolución, orientación o tamaño de voxel.
    resample_to_img() alinea el atlas al fMRI con interpolación por
    vecino más cercano (para preservar los IDs enteros de las ROIs).

    Parámetros
    ----------
    fmri_img   : Nifti1Image — imagen fMRI 4D del sujeto (nibabel)
    atlas_name : str         — nombre del atlas a usar. Opciones:
                               'cort-maxprob-thr25-2mm'  Harvard-Oxford cortical (default)
                               'sub-maxprob-thr25-2mm'   Harvard-Oxford subcortical
                               'AAL'                     Automated Anatomical Labeling
                               'destrieux_2009'          Destrieux

    Retorna
    -------
    atlas_data : array (X, Y, Z) — ID de región en cada voxel del espacio fMRI
    atlas_obj  : objeto Nilearn  — contiene atlas_obj.labels con nombres de regiones
    """
    ATLAS_LOADERS = {
        'schaefer_100'            : lambda: datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2),
        'cort-maxprob-thr25-2mm' : lambda: datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm'),
        'sub-maxprob-thr25-2mm'  : lambda: datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm'),
        'AAL'                    : lambda: datasets.fetch_atlas_aal(),
        'destrieux_2009'         : lambda: datasets.fetch_atlas_destrieux_2009(),
    }

    if atlas_name not in ATLAS_LOADERS:
        raise ValueError(
            f"Atlas '{atlas_name}' no reconocido. Opciones: {list(ATLAS_LOADERS.keys())}"
        )

    print(f'Cargando atlas: {atlas_name}...')
    atlas_obj = ATLAS_LOADERS[atlas_name]()
    atlas_img = atlas_obj.maps

    atlas_resampled = resample_to_img(
        atlas_img,
        fmri_img.slicer[:, :, :, 0],
        interpolation='nearest'
    )
    atlas_data = atlas_resampled.get_fdata()

    n_regiones = len(atlas_obj.labels)
    print(f'Atlas listo. Shape: {atlas_data.shape} | Regiones: {n_regiones}')
    return atlas_data, atlas_obj


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de señales ROI
# ─────────────────────────────────────────────────────────────────────────────

def extraer_senales_roi(fmri_data, atlas_data, atlas_obj, min_voxels=100):
    """
    Extrae la señal promedio de cada ROI del atlas sobre el volumen fMRI.
    Solo incluye ROIs con al menos min_voxels voxeles.

    Para cada ROI: identifica los voxeles con ese ID (máscara booleana),
    extrae sus señales temporales y las promedia → una señal por ROI.

    Parámetros
    ----------
    fmri_data  : array (X, Y, Z, T)
    atlas_data : array (X, Y, Z)     — IDs de ROI por voxel (de cargar_atlas)
    atlas_obj  : objeto Nilearn      — para acceder a atlas_obj.labels
    min_voxels : int                 — mínimo de voxeles por ROI (default 100)

    Retorna
    -------
    roi_signals   : array (T, n_rois) — señal promedio por ROI
    selected_rois : list[int]          — IDs de las ROIs incluidas
    roi_names     : list[str]          — nombres de las ROIs incluidas
    roi_sizes     : dict {rid: n}      — tamaño en voxeles de cada ROI
    """
    roi_ids   = [int(i) for i in np.unique(atlas_data) if i != 0]
    roi_sizes = {rid: int(np.sum(atlas_data == rid)) for rid in roi_ids}

    selected_rois = [rid for rid in roi_ids if roi_sizes[rid] >= min_voxels]
    excluidas     = len(roi_ids) - len(selected_rois)

    signals   = []
    roi_names = []

    for rid in selected_rois:
        mask   = atlas_data == rid
        signal = fmri_data[mask, :].mean(axis=0)
        signals.append(signal)

        roi_names.append(nombre_roi_atlas(atlas_obj, rid))

    roi_signals = np.array(signals).T  # (T, n_rois)

    print(f'ROIs incluidas : {len(selected_rois)} de {len(roi_ids)} '
          f'({excluidas} excluidas por < {min_voxels} voxeles)')
    print(f'Shape señales  : {roi_signals.shape}  (T x n_rois)')

    return roi_signals, selected_rois, roi_names, roi_sizes


# ─────────────────────────────────────────────────────────────────────────────
# Cache para visores interactivos
# ─────────────────────────────────────────────────────────────────────────────

def precalcular_cache_roi(fmri_data, atlas_data, atlas_obj, bandpass_fn, tr,
                          min_voxels=100):
    """
    Pre-calcula señal original y filtrada para cada ROI y las guarda en un
    diccionario. Necesario para los visores interactivos: sin el cache,
    cada clic tardaría varios segundos en recalcular.

    Parámetros
    ----------
    fmri_data   : array (X, Y, Z, T)
    atlas_data  : array (X, Y, Z)
    atlas_obj   : objeto Nilearn
    bandpass_fn : función — filtrado.bandpass_filter
    tr          : float  — TR del sujeto en segundos (usar get_tr() por sujeto)
    min_voxels  : int    — ROIs con menos voxeles no se cachean

    Retorna
    -------
    roi_signal_cache : dict {rid: (sig_orig, sig_filt)}
    roi_name_map     : dict {rid: nombre_str}
    roi_sizes        : dict {rid: n_voxeles}
    """
    roi_ids   = [int(i) for i in np.unique(atlas_data) if i != 0]
    roi_sizes = {rid: int(np.sum(atlas_data == rid)) for rid in roi_ids}

    roi_name_map     = {}
    roi_signal_cache = {}

    print(f'Pre-calculando cache de ROIs con TR={tr}s (puede tardar ~30s)...')

    for rid in roi_ids:
        roi_name_map[rid] = nombre_roi_atlas(atlas_obj, rid)

        if roi_sizes[rid] >= min_voxels:
            mask = atlas_data == rid
            sig  = fmri_data[mask, :].mean(axis=0)
            roi_signal_cache[rid] = (sig, bandpass_fn(sig, tr=tr))

    print(f'Cache listo: {len(roi_signal_cache)} ROIs precalculadas.')
    return roi_signal_cache, roi_name_map, roi_sizes
