# Datos locales

En Git solo se conserva el archivo fenotipico necesario para cruzar
`SUB_ID`, `FILE_ID`, `DX_GROUP` y `SITE_ID`.

Los NIfTI de ABIDE PCP, atlas descargados, archivos parciales y matrices
derivadas son artefactos locales pesados y estan excluidos mediante
`.gitignore`. La corrida principal completa se almacena fuera del repositorio,
idealmente bajo la ruta indicada por la variable de entorno
`ABIDE_DATA_ROOT`.

La carpeta `data/ABIDE_pcp/cpac/` puede contener una muestra local para pruebas,
pero no representa por si sola la cohorte de 723 sujetos usada en el paper.
