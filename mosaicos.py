import streamlit as st
import os
import pystac_client
import planetary_computer
import stackstac
import rioxarray
import folium
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from calendar import monthrange
from folium.plugins import Draw, LocateControl
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Generador de Mosaicos Satelitales", layout="wide", page_icon="🗺️")

# --- INICIALIZACIÓN DE ESTADOS ---
states = {
    "geotiff_data": None,
    "jpg_hd_data": None,
    "kmz_data": None,
    "preview_jpg": None,
    "search_results": None,
    "active_bbox": None,
    "area_km2": 0.0,
    "is_searching": False
}

for key, default in states.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
    <style>
    html, body, [class*="st-"] { font-size: 0.95rem !important; }
    .block-container { padding-top: 1.5rem !important; }
    h1 { font-size: 1.6rem !important; font-weight: 800 !important; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .highlight-box { border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: #f9f9f9; margin-bottom: 10px; }
    .stMultiSelect div[role="listbox"] { font-size: 0.8rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ Generador de Mosaicos (Sentinel-2 & Landsat)")

# --- CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "viz": {"Color Natural": ["B04", "B03", "B02"], "Falso Color (Agua)": ["B08", "B11", "B04"]},
        "res": 10, "cloud_key": "eo:cloud_cover", "scale": 1.0, "offset": 0.0
    },
    "Landsat 8/9": {
        "collection": "landsat-c2-l2",
        "viz": {"Color Natural": ["red", "green", "blue"], "Falso Color (Agua)": ["nir08", "swir16", "red"]},
        "res": 30, "cloud_key": "eo:cloud_cover", "scale": 0.0000275, "offset": -0.2
    }
}

# --- FUNCIONES CORE ---
def get_utm_epsg(lon, lat):
    utm_zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + utm_zone

def get_date_range(anio, mes_num):
    current_first = datetime(anio, mes_num, 1)
    prev_month_date = current_first - timedelta(days=1)
    start_date = datetime(prev_month_date.year, prev_month_date.month, 1)
    next_month_date = current_first + timedelta(days=32)
    last_day_next = monthrange(next_month_date.year, next_month_date.month)[1]
    end_date = datetime(next_month_date.year, next_month_date.month, last_day_next)
    return start_date, end_date

def compute_with_retry(xarray_obj, retries=3, delay=5):
    for i in range(retries):
        try:
            return xarray_obj.compute()
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
                continue
            else:
                raise e

def normalize_image(img_arr, scale=1.0, offset=0.0):
    rows, cols, bands = img_arr.shape
    out = np.zeros((rows, cols, bands), dtype=np.uint8)
    for i in range(bands):
        band = img_arr[:, :, i].astype(np.float32) * scale + offset
        # Máscara estricta para ignorar ceros y NaNs en el cálculo de histograma
        mask = np.isfinite(band) & (band > 0.0001)
        valid = band[mask]
        if valid.size > 100:
            vmin, vmax = np.percentile(valid, [2, 98])
            if vmax > vmin:
                norm_band = np.zeros_like(band, dtype=np.uint8)
                norm_band[mask] = np.clip((band[mask] - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
                out[:, :, i] = norm_band
        del band
    return out

def add_text_to_image(pil_img, text):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=int(pil_img.height * 0.025))
    except:
        font = ImageFont.load_default()
    w, h = pil_img.size
    margin = 10
    draw.text((margin + 2, h - margin - 32), text, fill="black", font=font)
    draw.text((margin, h - margin - 30), text, fill="white", font=font)
    return pil_img

def create_kmz(bounds, image_bytes):
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Folder>
    <name>Mosaico Satelital</name>
    <GroundOverlay>
      <name>Mosaico Procesado</name>
      <Icon><href>files/mosaic.jpg</href></Icon>
      <LatLonBox>
        <north>{bounds[3]}</north><south>{bounds[1]}</south><east>{bounds[2]}</east><west>{bounds[0]}</west>
      </LatLonBox>
    </GroundOverlay>
  </Folder>
</kml>"""
    kmz_buffer = io.BytesIO()
    with zipfile.ZipFile(kmz_buffer, "w") as kmz:
        kmz.writestr("doc.kml", kml_content)
        kmz.writestr("files/mosaic.jpg", image_bytes)
    return kmz_buffer.getvalue()

# --- 1. AOI SELECTION ---
st.subheader("1. Definir Área de Interés (Máx. 7000 km²)")
m = folium.Map(location=[-34.6, -58.4], zoom_start=10)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width='stretch', height=400, key="main_map")

if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
    st.session_state.active_bbox = [min(lons), min(lats), max(lons), max(lats)]
    st.session_state.area_km2 = (abs(st.session_state.active_bbox[2]-st.session_state.active_bbox[0])*111) * (abs(st.session_state.active_bbox[3]-st.session_state.active_bbox[1])*111)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Parámetros de Búsqueda")
    sat_choice = st.selectbox("Satélite", options=list(SAT_CONFIG.keys()))
    conf = SAT_CONFIG[sat_choice]
    viz_mode = st.radio("Modo Visual", options=list(conf["viz"].keys()))
    selected_assets = conf["viz"][viz_mode]
    st.markdown("---")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    c1, c2 = st.columns(2)
    with c1: mes_nombre = st.selectbox("Mes Central", meses, index=datetime.now().month - 1)
    with c2: anio = st.number_input("Año", min_value=1984, max_value=datetime.now().year, value=datetime.now().year)
    mes_num = meses.index(mes_nombre) + 1
    start_dt, end_dt = get_date_range(anio, mes_num)
    max_cloud = st.slider("Nubosidad Máx. (%)", 0, 100, 20)
    res_final = st.number_input("Resolución Salida (m)", value=conf["res"], min_value=10)

if st.session_state.active_bbox:
    if st.session_state.area_km2 > 7000:
        st.error(f"⚠️ Área demasiado grande: {st.session_state.area_km2:.1f} km².")
    else:
        st.success(f"📍 Área seleccionada: {st.session_state.area_km2:.1f} km².")

# --- 2. BÚSQUEDA Y SELECCIÓN ---
if st.session_state.active_bbox and st.session_state.area_km2 <= 7000:
    search_label = "⌛ Buscando..." if st.session_state.is_searching else "🔍 Buscar Imágenes"
    
    if st.button(search_label, disabled=st.session_state.is_searching):
        st.session_state.is_searching = True
        st.rerun()

    if st.session_state.is_searching:
        with st.spinner("Analizando catálogo y cobertura..."):
            try:
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                search = catalog.search(
                    collections=[conf["collection"]],
                    bbox=st.session_state.active_bbox,
                    datetime=f"{start_dt.isoformat()}/{end_dt.isoformat()}",
                    query={conf["cloud_key"]: {"lt": max_cloud}},
                    sortby=[{"field": "properties.datetime", "direction": "desc"}]
                )
                items = list(search.items())
                if items:
                    items = items[:30] # Limitamos para velocidad de análisis
                    epsg_check = get_utm_epsg((st.session_state.active_bbox[0]+st.session_state.active_bbox[2])/2, (st.session_state.active_bbox[1]+st.session_state.active_bbox[3])/2)
                    
                    # Chequeo rápido de cobertura para informar al usuario
                    check_stack = stackstac.stack(items, assets=[selected_assets[0]], bounds_latlon=st.session_state.active_bbox, epsg=epsg_check, resolution=1000)
                    check_data = check_stack.compute()
                    
                    df_items = []
                    for i, item in enumerate(items):
                        data = check_data[i].values
                        nodata_pct = (np.sum((data <= 0) | np.isnan(data)) / data.size) * 100
                        df_items.append({
                            "ID": item.id,
                            "Fecha": item.datetime.strftime('%d/%m/%Y'),
                            "Nubes": item.properties[conf['cloud_key']],
                            "Vacío": float(nodata_pct),
                            "Objeto": item
                        })
                    st.session_state.search_results = pd.DataFrame(df_items)
                else:
                    st.warning("No se encontraron imágenes.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                st.session_state.is_searching = False
                st.rerun()

    if st.session_state.search_results is not None:
        st.subheader("2. Componer Mosaico")
        
        # UI DE SELECCIÓN Y ESTRATEGIA
        col_ui1, col_ui2 = st.columns([0.7, 0.3])
        
        with col_ui1:
            options_map = {}
            for idx, row in st.session_state.search_results.iterrows():
                label = f"{row['Fecha']} | ☁️ {row['Nubes']:.1f}% | ⬛ Vacío: {row['Vacío']:.1f}%"
                if row['Vacío'] > 10: label += " ⚠️ (Parcial)"
                options_map[label] = row['ID']

            selected_labels = st.multiselect(
                "Selecciona y ordena las imágenes (la primera será la base):", 
                options=list(options_map.keys()),
                default=list(options_map.keys())[:3]
            )
        
        with col_ui2:
            mosaic_mode = st.selectbox("Estrategia de Mosaico", 
                ["Priorizar Selección (First)", "Mediana (Mejor para Nubes)", "Promedio (Mean)"],
                help="First: Usa la primera imagen y llena huecos con las siguientes. Mediana: Elimina nubes comparando píxeles.")

        # PROCESAMIENTO
        selected_ids = [options_map[lab] for lab in selected_labels]
        # Respetamos el orden de selección del usuario
        selected_items = []
        for sid in selected_ids:
            item_obj = st.session_state.search_results[st.session_state.search_results["ID"] == sid]["Objeto"].values[0]
            selected_items.append(item_obj)

        if selected_items:
            epsg_utm = get_utm_epsg((st.session_state.active_bbox[0]+st.session_state.active_bbox[2])/2, (st.session_state.active_bbox[1]+st.session_state.active_bbox[3])/2)
            col_pre1, col_pre2 = st.columns([0.65, 0.35])
            
            with col_pre1:
                if st.button("🖼️ Ver Previsualización del Mosaico"):
                    with st.spinner("Fusionando capas..."):
                        try:
                            # Forzamos fill_value=np.nan y filtramos ceros agresivamente
                            stack = stackstac.stack(selected_items, assets=selected_assets, bounds_latlon=st.session_state.active_bbox, 
                                                    epsg=epsg_utm, resolution=res_final*15, fill_value=np.nan)
                            
                            # MÁSCARA CRÍTICA: Tratar el negro (0) como NaN antes de fusionar
                            stack = stack.where(stack > 1) 
                            
                            if "Mediana" in mosaic_mode:
                                mosaic = stack.median(dim="time").compute()
                            elif "Promedio" in mosaic_mode:
                                mosaic = stack.mean(dim="time").compute()
                            else:
                                mosaic = stackstac.mosaic(stack).compute()
                            
                            mosaic = mosaic.astype(np.float32)
                            img_np = np.moveaxis(mosaic.values, 0, -1)
                            preview_img = normalize_image(img_np, conf["scale"], conf["offset"])
                            
                            pil_preview = Image.fromarray(preview_img)
                            dates_txt = f"Mosaico: {len(selected_items)} escenas | Modo: {mosaic_mode.split('(')[0]}"
                            pil_preview = add_text_to_image(pil_preview, dates_txt)
                            
                            st.image(pil_preview, caption="Previsualización del resultado final", width='stretch')
                            buf = io.BytesIO(); pil_preview.save(buf, format="JPEG")
                            st.session_state.preview_jpg = buf.getvalue()
                        except Exception as e:
                            st.error(f"Error al generar mosaico: {e}")

            with col_pre2:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.write("### 🚀 Exportación")
                w_tif = st.checkbox("GeoTIFF (UTM)", value=True)
                w_jpg = st.checkbox("JPG (Leyenda)", value=False)
                w_kmz = st.checkbox("KMZ (Google Earth)", value=False)
                
                if st.button("⚙️ Procesar en HD"):
                    with st.status("Ejecutando proceso HD...") as status:
                        try:
                            stack_hd = stackstac.stack(selected_items, assets=selected_assets, bounds_latlon=st.session_state.active_bbox, 
                                                       epsg=epsg_utm, resolution=res_final, fill_value=np.nan, chunksize=1024)
                            
                            # Filtro para eliminar bordes negros
                            stack_hd = stack_hd.where(stack_hd > 1)
                            
                            if "Mediana" in mosaic_mode:
                                mosaic_hd = stack_hd.median(dim="time").compute()
                            elif "Promedio" in mosaic_mode:
                                mosaic_hd = stack_hd.mean(dim="time").compute()
                            else:
                                mosaic_hd = stackstac.mosaic(stack_hd).compute()
                            
                            mosaic_hd = mosaic_hd.astype(np.float32)
                            
                            if w_tif:
                                status.update(label="Generando GeoTIFF...")
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                    mosaic_hd.rio.write_nodata(np.nan, inplace=True)
                                    mosaic_hd.rio.to_raster(tmp.name, compress='lzw', tiled=True)
                                    with open(tmp.name, 'rb') as f: st.session_state.geotiff_data = f.read()
                            
                            if w_jpg or w_kmz:
                                status.update(label="Normalizando imagen...")
                                img_hd_np = np.moveaxis(mosaic_hd.values, 0, -1)
                                norm_hd = normalize_image(img_hd_np, conf["scale"], conf["offset"])
                                pil_hd = Image.fromarray(norm_hd)
                                
                                if w_jpg:
                                    buf = io.BytesIO(); pil_hd.save(buf, format="JPEG", quality=95)
                                    st.session_state.jpg_hd_data = buf.getvalue()
                                
                                if w_kmz:
                                    status.update(label="Generando KMZ...")
                                    # KMZ usa resolución adaptativa para que Google Earth no colapse
                                    stack_kmz = stackstac.stack(selected_items, assets=selected_assets, bounds_latlon=st.session_state.active_bbox, epsg=4326, resolution=0.0004, fill_value=np.nan)
                                    stack_kmz = stack_kmz.where(stack_kmz > 1)
                                    mosaic_kmz = stackstac.mosaic(stack_kmz).compute().astype(np.float32)
                                    b4326 = mosaic_kmz.rio.bounds()
                                    img_kmz_np = np.moveaxis(mosaic_kmz.values, 0, -1)
                                    norm_kmz = normalize_image(img_kmz_np, conf["scale"], conf["offset"])
                                    pil_kmz = Image.fromarray(norm_kmz)
                                    buf_kmz = io.BytesIO(); pil_kmz.save(buf_kmz, format="JPEG", quality=85)
                                    st.session_state.kmz_data = create_kmz(b4326, buf_kmz.getvalue())
                            
                            status.update(label="✅ Mosaico completado", state="complete")
                        except Exception as e:
                            st.error(f"Error en proceso HD: {e}")

                if st.session_state.geotiff_data: st.download_button("📥 GeoTIFF", st.session_state.geotiff_data, "mosaico.tif", "image/tiff")
                if st.session_state.jpg_hd_data: st.download_button("📥 JPG HD", st.session_state.jpg_hd_data, "mosaico.jpg", "image/jpeg")
                if st.session_state.kmz_data: st.download_button("📥 KMZ", st.session_state.kmz_data, "mosaico_ge.kmz", "application/vnd.google-earth.kmz")
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Procesador de Mosaicos - Control de capas y eliminación de bordes negros activado.")