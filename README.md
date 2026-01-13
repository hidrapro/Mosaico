 Generador de Mosaicos Satelitales (Sentinel-2 & Landsat)

Esta aplicación de Streamlit permite buscar, previsualizar y procesar imágenes satelitales de las constelaciones Sentinel-2 y Landsat 8/9 utilizando el catálogo de Microsoft Planetary Computer. Está optimizada para crear mosaicos de grandes áreas (hasta 7000 km²) con exportación a formatos profesionales.

🚀 Características

Búsqueda Inteligente: Localiza escenas en el mes elegido, el anterior y el posterior.

Análisis de Cobertura: Calcula automáticamente el porcentaje de nubosidad y de área vacía (nodata) de cada escena antes de procesar.

Estrategias de Mosaico:

Priorizar Selección: Respeta el orden de las capas elegido por el usuario.

Promedio (Mean): Suaviza la transición entre imágenes.

Mediana (Median): Ideal para eliminar nubes y sombras de forma estadística.

Optimización de Memoria: Procesamiento en float32 y flujo de datos eficiente para evitar errores en áreas extensas.

Exportación Multiformato:

GeoTIFF: En coordenadas UTM con metadatos de NoData.

JPG HD: Con leyenda técnica integrada (fechas y satélite).

KMZ: Proyectado en WGS84 para visualización perfecta en Google Earth sin deformaciones.

🛠️ Instalación Local

Si deseas ejecutar este proyecto en tu computadora:

Clona el repositorio:

git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO


Crea un entorno virtual:

python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate


Instala las dependencias:

pip install -r requirements.txt


Ejecuta la aplicación:

streamlit run mosaicos.py


🌐 Despliegue en Streamlit Cloud

Para desplegar esta app:

Sube mosaico.py y requirements.txt a un repositorio de GitHub.

Conecta tu cuenta de GitHub en share.streamlit.io.

Selecciona el repositorio y lanza la aplicación asegurándote de que el "Main file path" sea mosaico.py.

Nota Técnica: El procesamiento de áreas cercanas al límite de 7000 km² requiere una conexión a internet estable debido a la gran cantidad de datos que se solicitan mediante el protocolo STAC.
