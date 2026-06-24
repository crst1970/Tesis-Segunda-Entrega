# Tesis Segunda Entrega - Pipeline ABIDE I

Pipeline reproducible para comparar conectividad cerebral entre sujetos TEA y
controles usando rs-fMRI de ABIDE I.

## Flujo principal

```text
ABIDE PCP / C-PAC / func_preproc / filt_noglobal
-> atlas Schaefer 2018 de 100 ROIs corticales
-> promedio BOLD por ROI
-> z-score por ROI (sin segundo band-pass local)
-> filtro de sujetos con T >= 146
-> conectividad por sujeto
   -> correlacion de Pearson
   -> Graphical Lasso, alpha fijo 0.5
   -> DirectLiNGAM, dirigido y normalizado por sujeto
-> vectorizacion
-> SVM RBF y Random Forest
-> validacion cruzada estratificada y agrupada por sitio
-> comparacion descriptiva Control vs TEA
```

La corrida reportada en el paper contiene 723 sujetos: 329 TEA y 394
controles, provenientes de 17 sitios. Antes del filtro temporal habia 871
sujetos; se excluyeron 148 con menos de 146 volumenes.

## Orden de ejecucion

1. `pipeline_v8.ipynb`
   - carga y valida la cohorte;
   - reutiliza o genera la parcelacion;
   - calcula matrices individuales;
   - vectoriza y clasifica;
   - guarda la trazabilidad y las metricas.
2. `pipeline_versiones9.ipynb`
   - carga exclusivamente los resultados de v8;
   - genera matrices promedio Control y TEA;
   - calcula diferencias TEA - Control;
   - produce heatmaps, conectomas top 20 y tablas interpretativas.
3. `Tesis_Entrega_2/main.tex`
   - consume las figuras y cifras reales de las dos etapas anteriores.

## Archivos canonicos

- `script/pipeline_abide.py`: orquestacion, validaciones, cache, vectorizacion
  y evaluacion.
- `script/parcelacion.py`: atlas y extraccion de senales ROI.
- `script/filtrado.py`: band-pass opcional y z-score.
- `script/conectividad.py`: Pearson, Graphical Lasso, LiNGAM y metodos
  experimentales.
- `pipeline_versiones9/analisis_control_tea.py`: analisis grupal y figuras.
- `tests/test_pipeline_smoke.py`: pruebas rapidas de interfaces y formas.

`script/clasificador.py` es un modulo legacy. No reproduce la validacion por
sitio ni las metricas del paper y no debe usarse para la corrida principal.

## Datos

El notebook busca primero la variable de entorno `ABIDE_DATA_ROOT`. Si no esta
definida, usa la ruta local historica `D:/Carpeta Cristian/Datos_tesis` cuando
existe y, como respaldo, la carpeta `data/` del repositorio.

Estructura esperada:

```text
ABIDE_DATA_ROOT/
└── ABIDE_pcp/
    ├── Phenotypic_V1_0b_preprocessed1.csv
    └── cpac/
        └── filt_noglobal/
            └── *_func_preproc.nii.gz
```

La corrida principal usa `filt_noglobal`; por tanto, el pipeline no aplica un
segundo band-pass local. Si se usa `nofilt_noglobal`, debe activarse el
band-pass local y regenerarse el cache ROI.

## Configuracion principal

- Atlas: `schaefer_100`.
- ROIs: 100, todas con al menos 100 voxeles tras remuestreo.
- Longitud temporal minima: 146 volumenes.
- Metodos: `pearson`, `graphical_lasso`, `lingam`.
- Graphical Lasso: `alpha=0.5`.
- Clasificadores: SVM RBF (`C=1`, pesos balanceados) y Random Forest
  (300 arboles, pesos balanceados).
- Evaluacion: 5 folds con `StratifiedGroupKFold`, agrupados por sitio.

La seleccion de `alpha=0.5` esto modificar xq quedo malito

LiNGAM devuelve una matriz dirigida. En esta implementacion, cada matriz
se divide por el mayor coeficiente absoluto del mismo sujeto. Por eso sus
valores son pesos relativos dentro de cada sujeto y no coeficientes causales
crudos comparables en escala absoluta entre sujetos.

## Ejecucion por linea de comandos

```powershell
python script/pipeline_abide.py `
  --source nifti_atlas `
  --fmri-dir "$env:ABIDE_DATA_ROOT\ABIDE_pcp\cpac\filt_noglobal" `
  --phenotypic "$env:ABIDE_DATA_ROOT\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv" `
  --output-dir resultados\pipeline_schaefer_100_all_valid_100rois_tp146 `
  --atlas-name schaefer_100 `
  --methods pearson graphical_lasso lingam `
  --classifiers svm rf `
  --all-available `
  --max-rois 100 `
  --min-timepoints 146 `
  --cv-strategy group_site `
  --graphical-lasso-alpha 0.5 `
  --skip-bandpass
```

## Dependencias y pruebas

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Los visores interactivos de las ultimas celdas de v8 requieren `ipywidgets` e
`ipympl`. No son necesarios para recalcular matrices, metricas o el paper.

## Resultados principales

La carpeta `resultados/` se excluye de Git por su tamano. Los archivos de mayor
trazabilidad son:

- `sujetos_incluidos.csv`
- `configuracion_pipeline.json`
- `trazabilidad_parcelacion_filtrado.csv`
- `resumen_timepoints_por_sitio_pre_filtro.csv`
- `resumen_timepoints_por_sitio.csv`
- `matrices/<metodo>_100rois/`
- `X_<metodo>.npy` y `y_<metodo>.npy`
- `tabla_comparativa_resultados.csv`
- `metricas_por_fold.csv`
- `predicciones_por_sujeto.csv`
- `resultados/pipeline_versiones9/figures/`
- `resultados/pipeline_versiones9/tables/`

Los CSV completos de aristas ocupan aproximadamente 1.8 GB y no son necesarios
para compilar el paper. Pueden regenerarse desde las matrices individuales.

## Alcance

Correlacion parcial, Granger y PCMCI permanecen como metodos experimentales y
no forman parte del paper ni de los resultados principales. LiNGAM debe
interpretarse como analisis dirigido exploratorio: la senal BOLD, la respuesta
hemodinamica y la ausencia de pruebas inferenciales impiden afirmar causalidad
clinica o biomarcadores validados.
