"""
clasificador.py
---------------
Clasificación ASD vs Control usando matrices de conectividad fMRI.

Pipeline completo:
  1. vectorizar()        : matriz (n_rois, n_rois) → vector de features
  2. construir_dataset() : múltiples sujetos → matriz X, vector y
  3. clasificar()        : entrena y evalúa SVM / Random Forest
  4. comparar_metodos()  : compara los 5 métodos de conectividad entre sí
  5. plot_resultados()   : visualiza accuracy, ROC, importancia de features

Métodos de conectividad soportados:
  Asociativos (simétricos):
  - 'pearson'   : correlación de Pearson
  - 'parcial'   : correlación parcial
  - 'gl'        : Graphical Lasso (sparse)

  Causales (asimétricos):
  - 'granger'   : Granger causality via VAR
  - 'pcmci'     : PCMCI causalidad condicional (requiere tigramite)

Clasificadores disponibles:
  - 'svm'  : SVM con kernel RBF  (recomendado para n_sujetos < 500)
  - 'rf'   : Random Forest       (recomendado cuando se quiere importancia de features)

Uso mínimo en el notebook
--------------------------
    from clasificador import construir_dataset, clasificar, comparar_metodos

    # Un método + un clasificador:
    X, y, feat_names = construir_dataset(sujetos, etiquetas, metodo='pearson')
    resultados = clasificar(X, y, modelo='svm')

    # Comparar todos los métodos con ambos clasificadores:
    comparar_metodos(datasets_dict, modelo='svm')
    comparar_metodos(datasets_dict, modelo='rf')
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     permutation_test_score)
from sklearn.metrics       import (roc_auc_score, roc_curve,
                                   confusion_matrix, ConfusionMatrixDisplay,
                                   classification_report)
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE


# ─────────────────────────────────────────────────────────────────────────────
# 1. VECTORIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def vectorizar(matrix, simetrica=True):
    """
    Convierte una matriz de conectividad en un vector de features.

    Para métodos simétricos (Pearson, parcial, GL): extrae el triángulo
    superior sin diagonal → n*(n-1)/2 features.
    Para métodos asimétricos (Granger): extrae todo excepto diagonal
    → n*(n-1) features.

    Parámetros
    ----------
    matrix    : array (n_rois, n_rois)
    simetrica : bool — True para Pearson/parcial/GL, False para Granger

    Retorna
    -------
    vector : array (n_features,)
    """
    n = matrix.shape[0]
    if simetrica:
        i, j = np.triu_indices(n, k=1)
    else:
        mask = ~np.eye(n, dtype=bool)
        return matrix[mask].astype(np.float32)
    return matrix[i, j].astype(np.float32)


def nombres_features(roi_names, simetrica=True):
    """
    Genera nombres legibles para cada feature del vector.

    Parámetros
    ----------
    roi_names : list[str] — nombres de las ROIs
    simetrica : bool

    Retorna
    -------
    list[str] — 'ROI_A — ROI_B' para cada par
    """
    n = len(roi_names)
    if simetrica:
        return [f'{roi_names[i]} — {roi_names[j]}'
                for i in range(n) for j in range(i+1, n)]
    else:
        return [f'{roi_names[i]} → {roi_names[j]}'
                for i in range(n) for j in range(n) if i != j]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCCIÓN DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def construir_dataset(lista_sujetos, etiquetas, metodo='pearson',
                      roi_names=None, verbose=True):
    """
    Construye la matriz de features X y el vector de etiquetas y
    procesando una lista de sujetos.

    Cada sujeto es un dict con las señales ROI ya extraídas y filtradas.
    Ver la sección de uso en el notebook para el formato exacto.

    Parámetros
    ----------
    lista_sujetos : list[dict]
        Cada elemento debe tener las claves:
          'roi_signals_z'  : array (T, n_rois) — señales filtradas + z-score
          'tr'             : float             — TR del sujeto
          Opcionalmente para Granger:
          'granger_matrix' : array (n_rois, n_rois) — pre-calculada

    etiquetas     : list[int] — 1=ASD, 0=Control (mismo orden que lista_sujetos)
    metodo        : str — 'pearson' | 'parcial' | 'gl' | 'granger'
    roi_names     : list[str] (opcional) — para nombrar features
    verbose       : bool

    Retorna
    -------
    X           : array (n_sujetos, n_features)
    y           : array (n_sujetos,)
    feat_names  : list[str] — nombres de las features
    """
    from conectividad import (correlacion, correlacion_parcial,
                               graphical_lasso, granger, pcmci)

    ES_SIMETRICO = {
        'pearson' : True,
        'parcial' : True,
        'gl'      : True,
        'granger' : False,
        'pcmci'   : False,
    }
    if metodo not in ES_SIMETRICO:
        raise ValueError(f"Método '{metodo}' no reconocido. Opciones: {list(ES_SIMETRICO.keys())}")

    simetrica = ES_SIMETRICO[metodo]
    vectores  = []
    y         = []

    for idx, (sujeto, label) in enumerate(zip(lista_sujetos, etiquetas)):
        t0  = time.time()
        sig = sujeto['roi_signals_z']

        if metodo == 'pearson':
            mat = correlacion(sig)
        elif metodo == 'parcial':
            mat = correlacion_parcial(sig)
        elif metodo == 'gl':
            mat, _ = graphical_lasso(sig)
        elif metodo == 'granger':
            # Si ya está pre-calculada (ahorra tiempo en loops)
            if 'granger_matrix' in sujeto:
                mat = sujeto['granger_matrix']
            else:
                mat, _ = granger(sig, maxlag=2, significance=0.05)
        elif metodo == 'pcmci':
            # Si ya está pre-calculada (recomendado: pcmci es costoso)
            if 'pcmci_matrix' in sujeto:
                mat = sujeto['pcmci_matrix']
            else:
                mat, _, _ = pcmci(sig, tau_max=2, pc_alpha=0.05)

        vec = vectorizar(mat, simetrica=simetrica)
        vectores.append(vec)
        y.append(label)

        if verbose:
            print(f'  Sujeto {idx+1}/{len(lista_sujetos)} — label={label} — '
                  f'{time.time()-t0:.1f}s — features={len(vec)}')

    X = np.array(vectores)
    y = np.array(y)

    n_asd  = np.sum(y == 1)
    n_ctrl = np.sum(y == 0)
    print(f'\nDataset listo: {X.shape}  |  ASD={n_asd}  Control={n_ctrl}')
    print(f'Features por sujeto: {X.shape[1]}')

    feat_names = None
    if roi_names is not None:
        feat_names = nombres_features(roi_names, simetrica=simetrica)

    return X, y, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASIFICADOR
# ─────────────────────────────────────────────────────────────────────────────

MODELOS = {
    'svm': SVC(
        kernel='rbf', C=1.0, probability=True, random_state=42
    ),
    'rf': RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
}


def clasificar(X, y, modelo='svm', n_splits=5, n_permutaciones=100,
               random_state=42, verbose=True):
    """
    Entrena y evalúa un clasificador con validación cruzada estratificada.

    Incluye:
      - Cross-validation (accuracy, AUC, F1) con StratifiedKFold
      - Test de permutación para verificar que el resultado no es azar
      - Reporte por fold

    Parámetros
    ----------
    X               : array (n_sujetos, n_features)
    y               : array (n_sujetos,)  — 1=ASD, 0=Control
    modelo          : str — 'svm' | 'rf'
                        'svm' : SVM RBF — mejor para datasets pequeños/medianos
                        'rf'  : Random Forest — permite analizar importancia de features
    n_splits        : int — folds de CV (default 5)
    n_permutaciones : int — permutaciones para test estadístico (default 100)
    random_state    : int
    verbose         : bool

    Retorna
    -------
    resultados : dict con claves:
        'accuracy_mean', 'accuracy_std',
        'auc_mean',      'auc_std',
        'f1_mean',       'f1_std',
        'p_value',       (permutation test)
        'cv_scores',     (scores por fold)
        'pipeline'       (pipeline entrenado en todos los datos)
    """
    if modelo not in MODELOS:
        raise ValueError(f"Modelo '{modelo}' no reconocido. Opciones: {list(MODELOS.keys())}")

    clf = MODELOS[modelo]

    # Pipeline: escalado + clasificador
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    clf)
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f'\n── Clasificador: {modelo.upper()}  |  CV: {n_splits}-fold ──')

    # Cross-validation
    cv_results = cross_validate(
        pipe, X, y, cv=cv,
        scoring={'accuracy': 'accuracy',
                 'auc':      'roc_auc',
                 'f1':       'f1'},
        return_train_score=True
    )

    acc_mean = cv_results['test_accuracy'].mean()
    acc_std  = cv_results['test_accuracy'].std()
    auc_mean = cv_results['test_auc'].mean()
    auc_std  = cv_results['test_auc'].std()
    f1_mean  = cv_results['test_f1'].mean()
    f1_std   = cv_results['test_f1'].std()

    if verbose:
        print(f'  Accuracy : {acc_mean:.3f} ± {acc_std:.3f}')
        print(f'  AUC-ROC  : {auc_mean:.3f} ± {auc_std:.3f}')
        print(f'  F1-score : {f1_mean:.3f} ± {f1_std:.3f}')
        print(f'  Train acc: {cv_results["train_accuracy"].mean():.3f} '
              f'(gap={cv_results["train_accuracy"].mean()-acc_mean:.3f})')

    # Permutation test
    print(f'  Permutation test ({n_permutaciones} permutaciones)...')
    score_obs, perm_scores, p_value = permutation_test_score(
        pipe, X, y, cv=cv, n_permutations=n_permutaciones,
        scoring='accuracy', random_state=random_state, n_jobs=-1
    )
    print(f'  p-value  : {p_value:.4f}  {"✓ significativo" if p_value < 0.05 else "✗ no significativo"}')

    # Entrenar en todos los datos para devolver pipeline final
    pipe_final = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    MODELOS[modelo])
    ])
    pipe_final.fit(X, y)

    return {
        'modelo'         : modelo,
        'accuracy_mean'  : acc_mean,
        'accuracy_std'   : acc_std,
        'auc_mean'       : auc_mean,
        'auc_std'        : auc_std,
        'f1_mean'        : f1_mean,
        'f1_std'         : f1_std,
        'p_value'        : p_value,
        'perm_scores'    : perm_scores,
        'score_obs'      : score_obs,
        'cv_scores'      : cv_results,
        'pipeline'       : pipe_final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARACIÓN DE MÉTODOS
# ─────────────────────────────────────────────────────────────────────────────

def comparar_metodos(datasets_dict, modelo='svm', n_splits=5):
    """
    Compara múltiples métodos de conectividad con el mismo clasificador.

    Parámetros
    ----------
    datasets_dict : dict {nombre_metodo: (X, y)}
        Ej: {'pearson': (X_p, y), 'parcial': (X_pc, y),
             'gl': (X_gl, y), 'granger': (X_g, y), 'pcmci': (X_pcmci, y)}
    modelo        : str — 'svm' | 'rf'  (usar ambos para comparar)
    n_splits      : int

    Retorna
    -------
    resumen : dict {nombre_metodo: resultados_dict}

    Notas
    -----
    Para una comparación completa, llamar dos veces:
        resumen_svm = comparar_metodos(datasets_dict, modelo='svm')
        resumen_rf  = comparar_metodos(datasets_dict, modelo='rf')
    """
    resumen = {}

    print(f'\n{"="*55}')
    print(f'  COMPARACIÓN DE MÉTODOS — clasificador: {modelo.upper()}')
    print(f'{"="*55}')

    for nombre, (X, y) in datasets_dict.items():
        print(f'\n[{nombre}]  shape={X.shape}')
        res = clasificar(X, y, modelo=modelo, n_splits=n_splits,
                         n_permutaciones=100, verbose=True)
        resumen[nombre] = res

    # Tabla resumen
    print(f'\n{"─"*65}')
    print(f'{"Método":<18} {"Accuracy":>10} {"AUC":>10} {"F1":>10} {"p-val":>10}')
    print(f'{"─"*65}')
    for nombre, res in resumen.items():
        print(f'{nombre:<18} '
              f'{res["accuracy_mean"]:.3f}±{res["accuracy_std"]:.3f}  '
              f'{res["auc_mean"]:.3f}±{res["auc_std"]:.3f}  '
              f'{res["f1_mean"]:.3f}±{res["f1_std"]:.3f}  '
              f'{res["p_value"]:.4f}')
    print(f'{"─"*65}')

    return resumen


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def plot_permutation_test(resultados, titulo=''):
    """
    Histograma del permutation test: distribución nula vs accuracy observado.

    Parámetros
    ----------
    resultados : dict — output de clasificar()
    titulo     : str
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(resultados['perm_scores'], bins=20, color='steelblue',
            alpha=0.75, label='Distribución nula (permutaciones)')
    ax.axvline(resultados['score_obs'], color='tomato', lw=2.5,
               label=f'Accuracy observado = {resultados["score_obs"]:.3f}')
    ax.axvline(resultados['perm_scores'].mean(), color='gray',
               lw=1.5, linestyle='--', label='Media nula')

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Permutation Test — {resultados["modelo"].upper()}'
                 + (f'\n{titulo}' if titulo else '')
                 + f'\np = {resultados["p_value"]:.4f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_roc_cv(X, y, modelo='svm', n_splits=5, titulo=''):
    """
    Curva ROC por fold de CV + media ± std.

    Parámetros
    ----------
    X       : array (n_sujetos, n_features)
    y       : array (n_sujetos,)
    modelo  : str
    n_splits: int
    titulo  : str
    """
    clf  = MODELOS[modelo]
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    tprs    = []
    aucs    = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 5))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        pipe.fit(X[train_idx], y[train_idx])
        prob = pipe.predict_proba(X[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[test_idx], prob)
        auc_fold    = roc_auc_score(y[test_idx], prob)
        aucs.append(auc_fold)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, alpha=0.3, lw=1,
                label=f'Fold {fold+1} (AUC={auc_fold:.2f})')

    mean_tpr    = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc    = np.mean(aucs)
    std_auc     = np.std(aucs)
    std_tpr     = np.std(tprs, axis=0)

    ax.plot(mean_fpr, mean_tpr, color='tomato', lw=2.5,
            label=f'Media (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    alpha=0.15, color='tomato')
    ax.plot([0,1], [0,1], 'k--', lw=1, label='Azar')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Curva ROC — {modelo.upper()}'
                 + (f'\n{titulo}' if titulo else ''))
    ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_comparacion_metodos(resumen_dict):
    """
    Barplot comparando accuracy y AUC entre métodos de conectividad.

    Parámetros
    ----------
    resumen_dict : dict — output de comparar_metodos()
    """
    nombres  = list(resumen_dict.keys())
    accs     = [resumen_dict[n]['accuracy_mean'] for n in nombres]
    acc_stds = [resumen_dict[n]['accuracy_std']  for n in nombres]
    aucs     = [resumen_dict[n]['auc_mean']      for n in nombres]
    auc_stds = [resumen_dict[n]['auc_std']       for n in nombres]
    pvals    = [resumen_dict[n]['p_value']        for n in nombres]

    x    = np.arange(len(nombres))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - ancho/2, accs, ancho, yerr=acc_stds,
                   label='Accuracy', color='steelblue',  alpha=0.85, capsize=4)
    bars2 = ax.bar(x + ancho/2, aucs, ancho, yerr=auc_stds,
                   label='AUC-ROC',  color='tomato', alpha=0.85, capsize=4)

    # Marcar significancia con asterisco
    for i, pval in enumerate(pvals):
        ymax = max(accs[i] + acc_stds[i], aucs[i] + auc_stds[i])
        marker = '***' if pval < 0.001 else ('**' if pval < 0.01 else
                 ('*' if pval < 0.05 else 'ns'))
        ax.text(i, ymax + 0.02, marker, ha='center', fontsize=10)

    ax.axhline(0.5, color='gray', lw=1.2, linestyle='--', alpha=0.6,
               label='Azar (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_ylim(0.3, 1.05)
    ax.set_title('Comparación de métodos de conectividad')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_comparacion_clasificadores(datasets_dict, n_splits=5):
    """
    Compara SVM vs RF en todos los métodos de conectividad.

    Genera una tabla y un barplot con accuracy y AUC para la combinación
    (método_conectividad × clasificador), totalizando 5×2 = 10 combinaciones.

    Parámetros
    ----------
    datasets_dict : dict {nombre_metodo: (X, y)}
    n_splits      : int

    Retorna
    -------
    resumen : dict {nombre_metodo: {'svm': res_svm, 'rf': res_rf}}
    """
    resumen = {}

    print(f'\n{"="*65}')
    print(f'  COMPARACIÓN MÉTODOS × CLASIFICADORES (SVM vs RF)')
    print(f'{"="*65}')

    for nombre, (X, y) in datasets_dict.items():
        resumen[nombre] = {}
        for modelo in ['svm', 'rf']:
            print(f'\n[{nombre}  ×  {modelo.upper()}]  shape={X.shape}')
            res = clasificar(X, y, modelo=modelo, n_splits=n_splits,
                             n_permutaciones=100, verbose=False)
            resumen[nombre][modelo] = res
            print(f'  Accuracy={res["accuracy_mean"]:.3f}±{res["accuracy_std"]:.3f}  '
                  f'AUC={res["auc_mean"]:.3f}±{res["auc_std"]:.3f}  '
                  f'p={res["p_value"]:.4f}')

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    print(f'\n{"─"*75}')
    print(f'{"Método":<14} {"SVM Acc":>10} {"SVM AUC":>10} {"RF Acc":>10} {"RF AUC":>10}')
    print(f'{"─"*75}')
    for nombre, clfs in resumen.items():
        s = clfs['svm']
        r = clfs['rf']
        print(f'{nombre:<14} '
              f'{s["accuracy_mean"]:.3f}±{s["accuracy_std"]:.3f}  '
              f'{s["auc_mean"]:.3f}±{s["auc_std"]:.3f}  '
              f'{r["accuracy_mean"]:.3f}±{r["accuracy_std"]:.3f}  '
              f'{r["auc_mean"]:.3f}±{r["auc_std"]:.3f}')
    print(f'{"─"*75}')

    # ── Barplot ────────────────────────────────────────────────────────────────
    nombres  = list(resumen.keys())
    x        = np.arange(len(nombres))
    ancho    = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, metrica, label in zip(
        axes,
        ['accuracy_mean', 'auc_mean'],
        ['Accuracy', 'AUC-ROC']
    ):
        svm_vals = [resumen[n]['svm'][metrica]    for n in nombres]
        svm_stds = [resumen[n]['svm'][metrica.replace('mean','std')] for n in nombres]
        rf_vals  = [resumen[n]['rf'][metrica]     for n in nombres]
        rf_stds  = [resumen[n]['rf'][metrica.replace('mean','std')]  for n in nombres]

        ax.bar(x - ancho/2, svm_vals, ancho, yerr=svm_stds,
               label='SVM', color='steelblue', alpha=0.85, capsize=4)
        ax.bar(x + ancho/2, rf_vals,  ancho, yerr=rf_stds,
               label='RF',  color='tomato',    alpha=0.85, capsize=4)
        ax.axhline(0.5, color='gray', lw=1.2, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, fontsize=9)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f'{label} — SVM vs RF')
        ax.legend(fontsize=9)

    plt.suptitle('Comparación Métodos de Conectividad × Clasificadores', fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()

    return resumen


    """
    Visualiza los sujetos en 2D usando PCA o t-SNE.
    Útil para ver si ASD y Control son separables en el espacio de features.

    Parámetros
    ----------
    X                : array (n_sujetos, n_features)
    y                : array (n_sujetos,)
    metodo_reduccion : str — 'pca' | 'tsne'
    titulo           : str
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    if metodo_reduccion == 'pca':
        reducer    = PCA(n_components=2, random_state=42)
        xlabel, ylabel = 'PC1', 'PC2'
    elif metodo_reduccion == 'tsne':
        reducer    = TSNE(n_components=2, perplexity=min(30, len(y)//2),
                          random_state=42)
        xlabel, ylabel = 't-SNE 1', 't-SNE 2'
    else:
        raise ValueError("metodo_reduccion debe ser 'pca' o 'tsne'")

    X_2d = reducer.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(6, 5))
    colores = {0: 'steelblue', 1: 'tomato'}
    labels  = {0: 'Control', 1: 'ASD'}

    for clase in [0, 1]:
        mask = y == clase
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colores[clase], label=labels[clase],
                   alpha=0.75, s=60, edgecolors='k', linewidths=0.4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Embedding {metodo_reduccion.upper()} — sujetos'
                 + (f'\n{titulo}' if titulo else ''))
    ax.legend()
    plt.tight_layout()
    plt.show()

    if metodo_reduccion == 'pca':
        var = reducer.explained_variance_ratio_
        print(f'Varianza explicada: PC1={var[0]:.1%}  PC2={var[1]:.1%}  '
              f'Total={sum(var):.1%}')


def plot_importancia_features(pipeline_entrenado, feat_names, top_n=20, titulo=''):
    """
    Muestra las features más importantes (solo para Random Forest).

    Parámetros
    ----------
    pipeline_entrenado : Pipeline — output de clasificar()['pipeline']
    feat_names         : list[str] — output de construir_dataset()
    top_n              : int — cuántas features mostrar
    titulo             : str
    """
    clf = pipeline_entrenado.named_steps['clf']

    if not hasattr(clf, 'feature_importances_'):
        print('plot_importancia_features solo disponible para Random Forest.')
        return

    importancias = clf.feature_importances_
    idx_top      = np.argsort(importancias)[::-1][:top_n]

    nombres_top = ([feat_names[i] for i in idx_top]
                   if feat_names else [f'feat_{i}' for i in idx_top])
    vals_top    = importancias[idx_top]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    ax.barh(range(top_n), vals_top[::-1], color='steelblue', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(nombres_top[::-1], fontsize=8)
    ax.set_xlabel('Importancia (Gini)')
    ax.set_title(f'Top {top_n} features — Random Forest'
                 + (f'\n{titulo}' if titulo else ''))
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6. UTILIDAD: construir dataset desde matrices ya calculadas
# ─────────────────────────────────────────────────────────────────────────────

def dataset_desde_matrices(matrices_list, etiquetas, simetrica=True):
    """
    Construye X directamente desde una lista de matrices ya calculadas.
    Útil cuando ya tenés las matrices guardadas y no querés recalcularlas.

    Parámetros
    ----------
    matrices_list : list[array (n_rois, n_rois)]
    etiquetas     : list[int] — 1=ASD, 0=Control
    simetrica     : bool

    Retorna
    -------
    X : array (n_sujetos, n_features)
    y : array (n_sujetos,)
    """
    X = np.array([vectorizar(m, simetrica=simetrica) for m in matrices_list])
    y = np.array(etiquetas)
    print(f'Dataset desde matrices: {X.shape}  |  '
          f'ASD={np.sum(y==1)}  Control={np.sum(y==0)}')
    return X, y