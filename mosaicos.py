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
from folium.plugins import Draw
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
import time
import warnings
from shapely.geometry import shape, box

# --- OPTIMIZACIÓN AGRESIVA DE ENTORNO PARA VELOCIDAD (GDAL) ---
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_COLUMN"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
os.environ["VSI_CACHE"] = "YES"
os.environ["GDAL_CACHEMAX"] = "512"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Generador de Mosaicos Satelitales", layout="wide", page_icon="🗺️")
warnings.filterwarnings('ignore', message='All-NaN slice encountered')

# --- INICIALIZACIÓN DE ESTADOS ---
states = {
    "geotiff_data": None,
    "jpg_hd_data": None,
    "kmz_data": None,
    "preview_jpg": None,
    "search_results": None,
    "active_bbox": None,
    "area_km2": 0.0,
    "is_searching": False,
    "suggestions": []
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
    .suggestion-card { border-left: 5px solid #ff4b4b; padding: 10px; background: #fff5f5; margin-bottom: 5px; border-radius: 0 5px 5px 0; font-size: 0.9rem; }
    .metric-card { background: #eef2f7; padding: 10px; border-radius: 5px; border-left: 4px solid #3498db; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ Generador de Mosaicos (Sentinel-2 & Landsat)")

# --- CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "viz": {"Falso Color (Agua)": ["B08", "B11", "B04"], "Color Natural": ["B04", "B03", "B02"]},
        "res": 10, "cloud_key": "eo:cloud_cover", "scale": 1.0, "offset": 0.0
    },
    "Landsat 8/9": {
        "collection": "landsat-c2-l2",
        "viz": {"Falso Color (Agua)": ["nir08", "swir16", "red"], "Color Natural": ["red", "green", "blue"]},
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

def compute_with_retry(xarray_obj, max_retries=3):
    for i in range(max_retries):
        try:
            return xarray_obj.compute()
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(1)
                continue
            else:
                raise e

def analyze_mosaic_holes(mosaic_data):
    vals = mosaic_data.values
    nodata_mask = np.isnan(vals) | (vals <= 0.0001)
    if len(nodata_mask.shape) == 3:
        holes = np.all(nodata_mask, axis=0) if nodata_mask.shape[0] < 10 else np.all(nodata_mask, axis=-1)
    else:
        holes = nodata_mask
    pct = (np.sum(holes) / holes.size) * 100
    return pct, holes

def suggest_fill_images(hole_mask, candidate_df, bbox):
    if candidate_df.empty or np.sum(hole_mask) == 0:
        return []
    suggestions = []
    aoi_geom = box(*bbox)
    for _, row in candidate_df.iterrows():
        item_geom = shape(row["Objeto"].geometry)
        intersection = aoi_geom.intersection(item_geom).area
        coverage_power = (intersection / aoi_geom.area) * 100
        if coverage_power > 1.0:
            suggestions.append({
                "ID": row["ID"], "Fecha": row["Fecha"], "Poder": coverage_power, "Nubes": row["Nubes"]
            })
    return sorted(suggestions, key=lambda x: x["Poder"], reverse=True)[:3]

def normalize_image(img_arr, scale=1.0, offset=0.0):
    rows, cols, bands = img_arr.shape
    out = np.zeros((rows, cols, bands), dtype=np.uint8)
    for i in range(bands):
        # Aplicamos escala y offset sobre los datos crudos
        band = img_arr[:, :, i].astype(np.float32) * scale + offset
        
        # Ignoramos NaNs y cualquier valor <= 0 (NoData de Landsat tras el offset es -0.2)
        mask = np.isfinite(band) & (band > 0.001)
        valid = band[mask]
        
        if valid.size > 100:
            sample = np.random.choice(valid, min(valid.size, 10000), replace=False)
            vmin, vmax = np.percentile(sample, [2, 98])
            
            if vmax <= vmin:
                vmax = vmin + 0.1
            
            norm_band = np.zeros_like(band, dtype=np.uint8)
            norm_band[mask] = np.clip((band[mask] - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            out[:, :, i] = norm_band
    return out

def add_text_to_image(pil_img, text):
    draw = ImageDraw.Draw(pil_img)
    try: font = ImageFont.truetype("arial.ttf", size=int(pil_img.height * 0.025))
    except: font = ImageFont.load_default()
    w, h = pil_img.size
    draw.text((12, h - 42), text, fill="black", font=font)
    draw.text((10, h - 40), text, fill="white", font=font)
    return pil_img

def create_kmz(bounds, image_bytes):
    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Folder>
    <name>Mosaico Satelital</name>
    <GroundOverlay>
      <name>Capa Satelital</name>
      <Icon><href>files/mosaic.jpg</href></Icon>
      <LatLonBox>
        <north>{bounds[3]}</north><south>{bounds[1]}</south><east>{bounds[2]}</east><west>{bounds[0]}</west>
      </LatLonBox>
    </GroundOverlay>
  </Folder>
</kml>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as kmz:
        kmz.writestr("doc.kml", kml_content)
        kmz.writestr("files/mosaic.jpg", image_bytes)
    return buf.getvalue()

# --- UI ---
st.subheader("1. Definir AOI (Área de Interés)")
m = folium.Map(location=[-34.6, -58.4], zoom_start=10)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width='stretch', height=400, key="main_map")

if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    st.session_state.active_bbox = [min(lons), min(lats), max(lons), max(lats)]
    st.session_state.area_km2 = (abs(st.session_state.active_bbox[2]-st.session_state.active_bbox[0])*111) * (abs(st.session_state.active_bbox[3]-st.session_state.active_bbox[1])*111)

with st.sidebar:
    st.header("⚙️ Configuración")
    
    if st.session_state.area_km2 > 0:
        st.markdown(f"""
            <div class="metric-card">
                <small>Área Seleccionada</small><br>
                <b>{st.session_state.area_km2:.2f} km²</b>
            </div>
        """, unsafe_allow_html=True)
        if st.session_state.area_km2 > 7000:
            st.error("Área demasiado grande (>7000 km²). Por favor, reduce el AOI.")

    sat_choice = st.selectbox("Satélite", options=list(SAT_CONFIG.keys()))
    conf = SAT_CONFIG[sat_choice]
    viz_mode = st.radio("Modo Visual", options=list(conf["viz"].keys()))
    selected_assets = conf["viz"][viz_mode]
    st.markdown("---")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    c1, c2 = st.columns(2)
    with c1: mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with c2: anio = st.number_input("Año", min_value=1984, max_value=datetime.now().year, value=datetime.now().year)
    start_dt, end_dt = get_date_range(anio, meses.index(mes_nombre) + 1)
    max_cloud = st.slider("Nubes máx (%)", 0, 100, 20)
    res_final = st.number_input("Resolución (m)", value=conf["res"], min_value=10)

# --- BÚSQUEDA ---
if st.session_state.active_bbox and st.session_state.area_km2 <= 7000:
    if st.button("🔍 Buscar Imágenes", disabled=st.session_state.is_searching):
        st.session_state.is_searching = True
        st.rerun()

    if st.session_state.is_searching:
        with st.spinner("Buscando en catálogo..."):
            try:
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                search = catalog.search(
                    collections=[conf["collection"]], 
                    bbox=st.session_state.active_bbox, 
                    datetime=f"{start_dt.isoformat()}/{end_dt.isoformat()}", 
                    query={conf["cloud_key"]: {"lt": max_cloud}}, 
                    sortby=[{"field": "properties.datetime", "direction": "desc"}]
                )
                items = list(search.items())[:30]
                if items:
                    aoi_poly = box(*st.session_state.active_bbox)
                    df = []
                    for item in items:
                        item_poly = shape(item.geometry)
                        coverage_pct = (aoi_poly.intersection(item_poly).area / aoi_poly.area) * 100
                        df.append({
                            "ID": item.id, "Fecha": item.datetime.strftime('%d/%m/%Y'), 
                            "Nubes": item.properties[conf['cloud_key']], 
                            "Vacío": float(100 - coverage_pct), "Objeto": item
                        })
                    st.session_state.search_results = pd.DataFrame(df)
                else: st.warning("Sin resultados.")
            except Exception as e: st.error(f"Error: {e}")
            finally: 
                st.session_state.is_searching = False
                st.rerun()

    if st.session_state.search_results is not None:
        st.subheader("2. Componer Mosaico")
        col_sel, col_est = st.columns([0.7, 0.3])
        with col_sel:
            opt = {f"{r['Fecha']} | ☁️ {r['Nubes']:.1f}% | ⬛ Vacío: {r['Vacío']:.1f}%": r['ID'] for _, r in st.session_state.search_results.iterrows()}
            selected_labels = st.multiselect("Prioridad de imágenes:", options=list(opt.keys()), default=list(opt.keys())[:3])
        with col_est:
            mosaic_mode = st.selectbox("Método Fusión", ["First (Rápido)", "Mediana (Limpio)", "Promedio"], index=0)

        selected_ids = [opt[l] for l in selected_labels]
        selected_items = [st.session_state.search_results[st.session_state.search_results["ID"] == sid]["Objeto"].values[0] for sid in selected_ids]

        if selected_items:
            epsg_utm = get_utm_epsg((st.session_state.active_bbox[0]+st.session_state.active_bbox[2])/2, (st.session_state.active_bbox[1]+st.session_state.active_bbox[3])/2)
            col_v1, col_v2 = st.columns([0.65, 0.35])
            with col_v1:
                if st.button("🖼️ Generar Vista Previa"):
                    with st.spinner("Procesando vista rápida..."):
                        try:
                            preview_res = max(res_final * 2, (abs(st.session_state.active_bbox[2] - st.session_state.active_bbox[0]) * 111000) / 700)
                            
                            stack = stackstac.stack(
                                selected_items, 
                                assets=selected_assets, 
                                bounds_latlon=st.session_state.active_bbox, 
                                epsg=epsg_utm, 
                                resolution=preview_res,
                                resampling=Resampling.nearest,
                                xy_coords=False,
                                fill_value=np.nan,
                                dtype="float64", # Cambiado a float64 para compatibilidad total con fill_value nan
                                rescale=False 
                            )
                            if "Mediana" in mosaic_mode: mosaic = compute_with_retry(stack.median(dim="time"))
                            elif "Promedio" in mosaic_mode: mosaic = compute_with_retry(stack.mean(dim="time"))
                            else: mosaic = compute_with_retry(stackstac.mosaic(stack))
                            
                            h_pct, h_mask = analyze_mosaic_holes(mosaic)
                            st.session_state.suggestions = suggest_fill_images(h_mask, st.session_state.search_results[~st.session_state.search_results["ID"].isin(selected_ids)], st.session_state.active_bbox) if h_pct > 1.0 else []

                            img = normalize_image(np.moveaxis(mosaic.values, 0, -1), conf["scale"], conf["offset"])
                            pil = add_text_to_image(Image.fromarray(img), f"Vista Rápida ({len(selected_items)} escenas)")
                            st.image(pil, width='stretch')
                            buf = io.BytesIO(); pil.save(buf, format="JPEG"); st.session_state.preview_jpg = buf.getvalue()
                        except Exception as e: st.error(f"Error de preview: {e}")

                if st.session_state.suggestions:
                    st.info("💡 Sugerencias para huecos:")
                    for sug in st.session_state.suggestions:
                        st.markdown(f'<div class="suggestion-card"><b>{sug["Fecha"]}</b> (Cubre +{sug["Poder"]:.1f}%)</div>', unsafe_allow_html=True)

            with col_v2:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.write("### 🚀 Exportación Full HD")
                w_tif = st.checkbox("GeoTIFF", True)
                w_jpg = st.checkbox("JPG HD", False)
                w_kmz = st.checkbox("KMZ (Google Earth)", False)
                
                if st.button("⚙️ Procesar Alta Resolución"):
                    with st.status("Procesando píxeles...") as status:
                        try:
                            stack_hd = stackstac.stack(selected_items, assets=selected_assets, bounds_latlon=st.session_state.active_bbox, epsg=epsg_utm, resolution=res_final, chunksize=1024, fill_value=np.nan, dtype="float64", rescale=False)
                            if "Mediana" in mosaic_mode: mosaic_hd = compute_with_retry(stack_hd.median(dim="time"))
                            elif "Promedio" in mosaic_mode: mosaic_hd = compute_with_retry(stack_hd.mean(dim="time"))
                            else: mosaic_hd = compute_with_retry(stackstac.mosaic(stack_hd))
                            
                            if w_tif:
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                    mosaic_hd.rio.to_raster(tmp.name, compress='lzw', tiled=True)
                                    with open(tmp.name, 'rb') as f: st.session_state.geotiff_data = f.read()
                            
                            if w_jpg or w_kmz:
                                norm_hd = normalize_image(np.moveaxis(mosaic_hd.values, 0, -1), conf["scale"], conf["offset"])
                                if w_jpg:
                                    b = io.BytesIO(); Image.fromarray(norm_hd).save(b, format="JPEG", quality=95); st.session_state.jpg_hd_data = b.getvalue()
                                
                                if w_kmz:
                                    stack_kmz = stackstac.stack(selected_items, assets=selected_assets, bounds_latlon=st.session_state.active_bbox, epsg=4326, resolution=res_final/111000, fill_value=np.nan, dtype="float64", rescale=False)
                                    if "Mediana" in mosaic_mode: m_kmz = compute_with_retry(stack_kmz.median(dim="time"))
                                    elif "Promedio" in mosaic_mode: m_kmz = compute_with_retry(stack_kmz.mean(dim="time"))
                                    else: m_kmz = compute_with_retry(stackstac.mosaic(stack_kmz))
                                    
                                    n_kmz = normalize_image(np.moveaxis(m_kmz.values, 0, -1), conf["scale"], conf["offset"])
                                    b_k = io.BytesIO(); Image.fromarray(n_kmz).save(b_k, format="JPEG", quality=85)
                                    st.session_state.kmz_data = create_kmz(m_kmz.rio.bounds(), b_k.getvalue())

                            status.update(label="✅ Archivos Listos", state="complete")
                        except Exception as e: st.error(f"Error HD: {e}")
                
                if st.session_state.geotiff_data: st.download_button("📥 GeoTIFF", st.session_state.geotiff_data, "mosaico.tif")
                if st.session_state.jpg_hd_data: st.download_button("📥 JPG HD", st.session_state.jpg_hd_data, "mosaico.jpg")
                if st.session_state.kmz_data: st.download_button("📥 KMZ", st.session_state.kmz_data, "mosaico.kmz")
                st.markdown('</div>', unsafe_allow_html=True)
else: st.info("👈 Dibuja un rectángulo en el mapa para comenzar.")