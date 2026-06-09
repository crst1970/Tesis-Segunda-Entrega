"""
visor.py
--------
Widgets interactivos para explorar datos fMRI.

Requiere %matplotlib widget al inicio del notebook.

Funciones:
  - visor_cortes          : navega cortes axiales con slider
  - visor_senal_voxel     : clic en voxel → señal original + filtrada
  - visor_parcelacion     : clic en ROI del atlas → nombre + señal
  - visor_conectividad    : clic en ROI → mapa de correlación 3D
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# ─────────────────────────────────────────────────────────────────────────────
# 1. VISOR DE CORTES
# ─────────────────────────────────────────────────────────────────────────────

def visor_cortes(fmri_data):
    """
    Navega cortes axiales y timepoints con dos sliders.

    Parámetros
    ----------
    fmri_data : array (X, Y, Z, T)
    """
    x_dim, y_dim, z_dim, t_dim = fmri_data.shape

    def mostrar(z, t):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(fmri_data[:, :, z, t].T, cmap='gray', origin='lower')
        ax.set_title(f'Corte axial z={z}, tiempo={t}')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    widgets.interact(
        mostrar,
        z=widgets.IntSlider(value=z_dim // 2, min=0, max=z_dim - 1, step=1, description='z'),
        t=widgets.IntSlider(value=0,           min=0, max=t_dim - 1, step=1, description='t')
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. VISOR SEÑAL VOXEL (original + filtrada al hacer clic)
# ─────────────────────────────────────────────────────────────────────────────

def visor_senal_voxel(fmri_data, bandpass_fn, tr):
    """
    Clic en un voxel del corte axial → muestra señal original y filtrada.

    Parámetros
    ----------
    fmri_data   : array (X, Y, Z, T)
    bandpass_fn : función de filtrado.bandpass_filter
    tr          : float — Repetition Time en segundos
    """
    x_dim, y_dim, z_dim, t_dim = fmri_data.shape

    def mostrar(z, t=0):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].imshow(fmri_data[:, :, z, t].T, cmap='gray', origin='lower')
        axes[0].set_title(f'Corte axial z={z}  |  Haz clic')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[1].set_title('Señal original')
        axes[1].set_xlabel('Tiempo (volúmenes)')
        axes[1].set_ylabel('Amplitud')
        axes[2].set_title('Señal filtrada (0.01–0.1 Hz)')
        axes[2].set_xlabel('Tiempo (volúmenes)')
        axes[2].set_ylabel('Amplitud')

        def onclick(event):
            if event.inaxes != axes[0] or event.xdata is None:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            if not (0 <= x < x_dim and 0 <= y < y_dim):
                return

            sig_orig = fmri_data[x, y, z, :]
            sig_filt = bandpass_fn(sig_orig, tr=tr)

            axes[0].cla()
            axes[0].imshow(fmri_data[:, :, z, t].T, cmap='gray', origin='lower')
            axes[0].scatter(x, y, c='red', s=60, zorder=5)
            axes[0].set_title(f'z={z}  —  voxel ({x}, {y}, {z})')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')

            axes[1].cla()
            axes[1].plot(sig_orig, color='steelblue', lw=1.2)
            axes[1].set_title(f'Original — ({x}, {y}, {z})')
            axes[1].set_xlabel('Tiempo (volúmenes)')
            axes[1].set_ylabel('Amplitud')

            axes[2].cla()
            axes[2].plot(sig_filt, color='tomato', lw=1.2)
            axes[2].set_title('Filtrada (0.01–0.1 Hz)')
            axes[2].set_xlabel('Tiempo (volúmenes)')
            axes[2].set_ylabel('Amplitud')

            fig.canvas.draw_idle()
            print(f'Voxel ({x}, {y}, {z})  std orig={sig_orig.std():.2f}  std filt={sig_filt.std():.2f}')

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

    widgets.interact(
        mostrar,
        z=widgets.IntSlider(value=z_dim // 2, min=0, max=z_dim - 1, step=1, description='z'),
        t=widgets.fixed(0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISOR PARCELACIÓN (clic en ROI → nombre + señal)
# ─────────────────────────────────────────────────────────────────────────────

def visor_parcelacion(fmri_data, atlas_data, roi_signal_cache, roi_name_map, roi_ids_all, roi_sizes_all, z_init=None, alpha_atlas=0.45):
    """
    Overlay del atlas sobre el fMRI. Clic en una región → nombre y señal.

    Parámetros
    ----------
    fmri_data        : array (X, Y, Z, T)
    atlas_data       : array (X, Y, Z)
    roi_signal_cache : dict {rid: (sig_orig, sig_filt)}   — de parcelacion.precalcular_cache_roi
    roi_name_map     : dict {rid: nombre}
    roi_ids_all      : list[int]
    roi_sizes_all    : dict {rid: n_voxeles}
    z_init           : int (opcional)
    alpha_atlas      : float — opacidad inicial del overlay
    """
    x_dim, y_dim, z_dim, t_dim = fmri_data.shape
    MIN_VOXELS = 100

    if z_init is None:
        z_init = z_dim // 2

    n_rois   = len(roi_ids_all)
    cmap_roi = plt.cm.get_cmap('tab20', n_rois)
    state    = {'z': z_init, 'alpha': alpha_atlas, 'roi_sel': None}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def draw_brain():
        z, alpha, roi_s = state['z'], state['alpha'], state['roi_sel']
        axes[0].cla()
        axes[0].imshow(fmri_data[:, :, z, 0].T, cmap='gray', origin='lower')

        atlas_slice = atlas_data[:, :, z].T
        overlay     = np.zeros((*atlas_slice.shape, 4))

        for idx, rid in enumerate(roi_ids_all):
            mask_2d = atlas_slice == rid
            if not mask_2d.any():
                continue
            r, g, b, _ = cmap_roi(idx)
            a = min(alpha * 1.9, 1.0) if rid == roi_s else alpha
            overlay[mask_2d] = [r, g, b, a]

        axes[0].imshow(overlay, origin='lower')

        if roi_s is not None:
            mask_2d = atlas_slice == roi_s
            if mask_2d.any():
                axes[0].contour(mask_2d, levels=[0.5], colors='white', linewidths=1.5)

        nombre = roi_name_map.get(roi_s, '') if roi_s else ''
        titulo = f'z={z}  —  {nombre}' if nombre else f'z={z}  ←  haz clic para identificar región'
        axes[0].set_title(titulo, fontsize=10)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')

    def draw_signal(rid):
        axes[1].cla()
        nombre = roi_name_map.get(rid, f'ROI {rid}')
        nvox   = roi_sizes_all.get(rid, 0)

        if rid not in roi_signal_cache:
            axes[1].set_title(f'{nombre}\nROI {rid} — muy pequeña (< {MIN_VOXELS} voxeles)', fontsize=9)
            return

        sig_orig, sig_filt = roi_signal_cache[rid]
        axes[1].plot(sig_orig, color='steelblue', lw=1.2, alpha=0.7, label='Original')
        axes[1].plot(sig_filt, color='tomato',    lw=1.4, alpha=0.9, label='Filtrada (0.01–0.1 Hz)')
        axes[1].set_title(f'{nombre}\nROI {rid}  —  {nvox} voxeles', fontsize=9)
        axes[1].set_xlabel('Tiempo (volúmenes)')
        axes[1].set_ylabel('Amplitud')
        axes[1].legend(fontsize=8)

    def on_click(event):
        if event.inaxes != axes[0] or event.xdata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        z = state['z']
        if not (0 <= x < x_dim and 0 <= y < y_dim):
            return

        rid = int(atlas_data[x, y, z])
        if rid == 0:
            print(f'Clic en ({x}, {y}, {z}) — fuera del atlas (fondo)')
            return

        state['roi_sel'] = rid
        nombre = roi_name_map.get(rid, f'ROI {rid}')
        nvox   = roi_sizes_all.get(rid, 0)
        print(f'ROI {rid}: {nombre}  |  {nvox} voxeles  |  voxel ({x}, {y}, {z})')

        draw_brain()
        draw_signal(rid)
        fig.canvas.draw_idle()

    sl_z = widgets.IntSlider(
        value=z_init, min=0, max=z_dim - 1, step=1,
        description='z (corte):',
        style={'description_width': 'auto'},
        layout=widgets.Layout(width='420px')
    )
    sl_alpha = widgets.FloatSlider(
        value=alpha_atlas, min=0.0, max=1.0, step=0.05,
        description='Opacidad atlas:',
        style={'description_width': 'auto'},
        layout=widgets.Layout(width='420px')
    )

    def on_z(change):
        state['z'] = change['new']
        draw_brain()
        fig.canvas.draw_idle()

    def on_alpha(change):
        state['alpha'] = change['new']
        draw_brain()
        fig.canvas.draw_idle()

    sl_z.observe(on_z,     names='value')
    sl_alpha.observe(on_alpha, names='value')
    fig.canvas.mpl_connect('button_press_event', on_click)

    draw_brain()
    axes[1].set_title('Haz clic en una región del atlas')
    plt.tight_layout()
    plt.show()
    display(widgets.VBox([sl_z, sl_alpha]))


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISOR CONECTIVIDAD (clic en ROI → mapa de correlación 3D)
# ─────────────────────────────────────────────────────────────────────────────

def visor_conectividad(fmri_data, atlas_data, roi_signal_cache, roi_name_map, roi_ids_all, z_init=None, alpha=0.3):
    """
    Clic en una ROI del atlas → calcula mapa de correlación 3D y lo muestra.

    Parámetros
    ----------
    fmri_data        : array (X, Y, Z, T)
    atlas_data       : array (X, Y, Z)
    roi_signal_cache : dict {rid: (sig_orig, sig_filt)}
    roi_name_map     : dict {rid: nombre}
    roi_ids_all      : list[int]
    z_init           : int (opcional)
    alpha            : float — opacidad overlay atlas
    """
    x_dim, y_dim, z_dim, t_dim = fmri_data.shape

    if z_init is None:
        z_init = z_dim // 2

    n_rois   = len(roi_ids_all)
    cmap_roi = plt.cm.get_cmap('tab20', n_rois)
    state    = {'z': z_init, 'corr_3D': None, 'roi_sel': None}

    def calcular_mapa_3D(seed_signal):
        n_voxels    = x_dim * y_dim * z_dim
        all_signals = fmri_data.reshape(n_voxels, t_dim)
        seed_norm   = seed_signal - seed_signal.mean()
        seed_std    = seed_signal.std()
        all_mean    = all_signals.mean(axis=1, keepdims=True)
        all_std     = all_signals.std(axis=1)
        valid       = all_std > 0
        corr_flat   = np.zeros(n_voxels)
        corr_flat[valid] = (
            ((all_signals[valid] - all_mean[valid]) @ seed_norm)
            / (all_std[valid] * seed_std * t_dim)
        )
        return corr_flat.reshape(x_dim, y_dim, z_dim)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mapa_vacio = np.zeros((x_dim, y_dim))
    im_corr    = axes[1].imshow(mapa_vacio.T, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    cbar       = fig.colorbar(im_corr, ax=axes[1])
    cbar.set_label('Correlación')

    def draw_atlas():
        z = state['z']
        axes[0].cla()
        axes[0].imshow(fmri_data[:, :, z, 0].T, cmap='gray', origin='lower')

        atlas_slice = atlas_data[:, :, z].T
        overlay     = np.zeros((*atlas_slice.shape, 4))

        for idx, rid in enumerate(roi_ids_all):
            mask_2d = atlas_slice == rid
            if not mask_2d.any():
                continue
            r, g, b, _ = cmap_roi(idx)
            a = min(alpha * 2, 1.0) if rid == state['roi_sel'] else alpha
            overlay[mask_2d] = [r, g, b, a]

        axes[0].imshow(overlay, origin='lower')
        nombre = roi_name_map.get(state['roi_sel'], '') if state['roi_sel'] else ''
        titulo = f"z={z}  —  {nombre}" if nombre else f"z={z}  ←  clic para seleccionar ROI"
        axes[0].set_title(titulo, fontsize=10)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')

    def draw_corr():
        z = state['z']
        axes[1].cla()
        if state['corr_3D'] is None:
            im2 = axes[1].imshow(np.zeros((x_dim, y_dim)).T, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
            axes[1].set_title('Haz clic en una ROI')
        else:
            im2 = axes[1].imshow(state['corr_3D'][:, :, z].T, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
            nombre = roi_name_map.get(state['roi_sel'], '')
            axes[1].set_title(f"Conectividad desde: {nombre}\nz={z}", fontsize=9)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        cbar.update_normal(im2)

    def onclick(event):
        if event.inaxes != axes[0] or event.xdata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        z = state['z']
        if not (0 <= x < x_dim and 0 <= y < y_dim):
            return

        rid = int(atlas_data[x, y, z])
        if rid == 0:
            print('Clic fuera del atlas (fondo)')
            return
        if rid not in roi_signal_cache:
            print(f'ROI {rid} muy pequeña, sin señal precalculada')
            return

        state['roi_sel'] = rid
        nombre = roi_name_map.get(rid, f'ROI {rid}')
        print(f'Semilla: {nombre} (ROI {rid}) — calculando mapa 3D...')

        _, sig_filt = roi_signal_cache[rid]
        state['corr_3D'] = calcular_mapa_3D(sig_filt)

        print('Listo.')
        draw_atlas()
        draw_corr()
        fig.canvas.draw_idle()

    sl_z = widgets.IntSlider(
        value=z_init, min=0, max=z_dim - 1, step=1,
        description='z (corte):',
        style={'description_width': 'auto'},
        layout=widgets.Layout(width='420px')
    )

    def on_z(change):
        state['z'] = change['new']
        draw_atlas()
        draw_corr()
        fig.canvas.draw_idle()

    sl_z.observe(on_z, names='value')
    fig.canvas.mpl_connect('button_press_event', onclick)

    draw_atlas()
    draw_corr()
    plt.tight_layout()
    plt.show()
    display(widgets.VBox([sl_z]))
