# Pipeline ABIDE fMRI con Schaefer 100 y filtro temporal

Este pipeline clasifica TEA/ASD vs Control usando ABIDE PCP y un atlas unico por corrida. La configuracion principal actual usa `schaefer_100` con `max_rois=100` y `min_timepoints=146` para excluir sujetos con series temporales demasiado cortas para metodos multivariados/direccionales.

```text
ABIDE func_preproc
-> Atlas Schaefer 100
-> Filtro temporal: timepoints >= 146
-> Senales ROI
-> Conectividad
   -> Analisis principal: Pearson, Graphical Lasso, LiNGAM
   -> Implementados pero excluidos por ahora: correlacion parcial, Granger, PCMCI
-> Vectorizacion
-> Clasificacion: SVM, Random Forest
-> Evaluacion, predicciones y visualizaciones
```

## Separacion v8/v9

- `pipeline_v8.ipynb`: calcula matrices, vectoriza y clasifica TEA vs Control con SVM y Random Forest.
- `pipeline_versiones9.ipynb`: usa las matrices generadas por v8 para el analisis Control vs TEA: matrices promedio, diferencia TEA-Control, magnitud, sparsity y conexiones mas diferentes.

## Politica de atlas

El proyecto usa atlas local sobre NIfTI, no ROIs precomputadas de ABIDE PCP. La corrida actual usa `schaefer_100` con 100 ROIs y filtra sujetos con menos de 146 timepoints; `craddock_cc200` y `destrieux_2009` siguen disponibles para comparaciones controladas si se ejecutan en carpetas de salida separadas.

## Ejecucion recomendada

Si los datos ya estan en el disco duro, no los descargues de nuevo. La configuracion actual de `pipeline_v8.ipynb` busca primero:

```text
D:/Carpeta Cristian/Datos_tesis/ABIDE_pcp/cpac/filt_noglobal
```

En esa carpeta ABIDE PCP ya aplico band-pass, por eso `pipeline_v8.ipynb` deja `APPLY_BANDPASS=False` automaticamente.

Si necesitas descargar desde cero en otra maquina, desde la raiz de `Tesis Segunda Entrega` puedes usar:

```powershell
python script/descargar_abide.py `
  --derivative func_preproc `
  --all-available
```

Si usas archivos en `filt_noglobal`, ABIDE PCP ya aplico band-pass; usa `--skip-bandpass` o `apply_bandpass=False` para evitar doble filtrado:

```powershell
python script/pipeline_abide.py `
  --source nifti_atlas `
  --fmri-dir "D:\Carpeta Cristian\Datos_tesis\ABIDE_pcp\cpac\filt_noglobal" `
  --phenotypic "D:\Carpeta Cristian\Datos_tesis\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv" `
  --output-dir resultados\pipeline_schaefer_100_all_valid_100rois_tp146 `
  --atlas-name schaefer_100 `
  --methods pearson graphical_lasso lingam `
  --classifiers svm rf `
  --all-available `
  --max-rois 100 `
  --min-timepoints 146 `
  --cv-strategy group_site `
  --granger-lag-strategy min_q `
  --maxlag 1 `
  --tau-max 1 `
  --skip-bandpass
```

## Validaciones

El pipeline valida atlas, orden de ROIs, shapes de senales, shapes de matrices, simetria en asociativos, diagonal/direccion en causales, numero de features, balance ASD/Control y particiones de validacion cruzada.

## Salidas principales

- `sujetos_incluidos.csv`: sujetos usados, etiqueta, sitio, timepoints y ROIs.
- `resumen_timepoints_por_sitio_pre_filtro.csv`: distribucion de timepoints antes del filtro temporal.
- `resumen_timepoints_por_sitio.csv`: distribucion de timepoints por sitio despues del filtro, ASD/Control y alertas `T <= ROIs`.
- `trazabilidad_parcelacion_filtrado.csv`: origen, atlas y etapa de filtrado.
- `parcelacion_atlas/`: atlas remuestreado, senales ROI y metadatos de parcelacion.
- `filtrado_zscore/*.npy`: senales ROI estandarizadas por sujeto.
- `matrices/<metodo>_<n>rois/*.npy`: matrices por sujeto y metodo.
- `feature_maps/<metodo>_feature_map.csv`: orden exacto de features.
- `X_<metodo>.npy` y `y_<metodo>.npy`: dataset vectorizado.
- `conectividad_tablas/<metodo>_edges.csv`: conexiones ROI-ROI por sujeto.
- `predicciones_por_sujeto.csv`: prediccion out-of-fold por sujeto.
- `metricas_por_fold.csv`: accuracy, AUC, F1, precision y recall por fold/sitio.
- `resumen_matrices_conectividad.csv`: densidad, sparsity, rango y simetria por matriz.
- `tabla_comparativa_resultados.csv`: metricas out-of-fold globales y desviacion estandar por fold, incluyendo `balanced_accuracy` y `specificity` para cohortes no balanceadas.
- `configuracion_pipeline.json`: parametros de ejecucion.

## Uso desde notebook

```python
tabla = pipeline_abide.run_pipeline(
    source="nifti_atlas",
    fmri_dir=DATA_DIR,
    phenotypic_csv=PHENO,
    output_dir=OUT_MAIN,
    atlas_name="schaefer_100",
    methods=["pearson", "graphical_lasso", "lingam"],
    classifiers=["svm", "rf"],
    apply_bandpass=False,
    cv_strategy="group_site",
    granger_lag_strategy="min_q",
    maxlag=1,
    tau_max=1,
    all_available=True,
    max_rois=100,
    min_timepoints=146,
)
```

## Notas conceptuales

Correlacion parcial, Granger y PCMCI siguen implementados, pero quedan comentados/excluidos del analisis principal actual. Si se usan, deben reportarse como analisis exploratorios o historicos en carpetas separadas.

Granger usa por defecto `min_q`: corrige todos los tests par-lag con FDR-BH y despues conserva el lag con menor q-value. Evita reportar como robustas conexiones que solo aparecen por escoger el menor p-value antes de corregir.

Schaefer 100 con `min_timepoints=146` conserva 100 ROIs, pero elimina sujetos cuya serie temporal es demasiado corta. Esto mejora la relacion entre observaciones temporales y variables cerebrales para Graphical Lasso y LiNGAM. Sigue siendo un compromiso, no una garantia causal: con fMRI BOLD los metodos direccionales deben interpretarse con prudencia.

Craddock CC200 puede usarse como analisis secundario de mayor resolucion, especialmente para metodos asociativos. Para causales, CC200 completo debe tratarse como experimento exploratorio por costo e inestabilidad.

La parcelacion se cachea por sujeto. Al volver a ejecutar, el pipeline reutiliza `filtrado_zscore/<file_id>_roi_signals_z.npy` y `parcelacion_atlas/<file_id>/roi_metadata.json` cuando el atlas coincide, evitando reabrir el NIfTI y remuestrear el atlas para sujetos ya procesados.

Si falla la descarga `craddock_2011_parcellations.tar.gz` desde Nilearn/NITRC, el codigo intenta automaticamente un fallback al atlas `cc200_roi_atlas.nii.gz` de ABIDE/FCP-INDI. Si tampoco hay internet, descarga manualmente `https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE/Resources/cc200_roi_atlas.nii.gz` y dejalo en `~/nilearn_data/abide_cc200/cc200_roi_atlas.nii.gz`.

LiNGAM se incluye para comparacion causal exploratoria, pero sus supuestos deben declararse: linealidad, no gaussianidad e independencia de errores. En BOLD/fMRI su interpretacion debe ser prudente por hemodinamica, bajo numero de timepoints y ruido. El codigo ahora falla de forma explicita si `T <= ROIs`; para LiNGAM usa `--max-rois`, un atlas de menor dimension o reduccion PCA/ICA previa.
