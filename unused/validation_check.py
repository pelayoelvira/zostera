import geopandas as gpd
import rasterio

# Ruta al GeoJSON y al TIFF
geojson_path = 'Data/1_groundtruth.geojson'
tiff_path = '2_villaviciosa_IZQ1.tif'

# Cargar GeoJSON
geojson_data = gpd.read_file(geojson_path)
print("CRS del GeoJSON:", geojson_data.crs)

# Cargar TIFF
with rasterio.open(tiff_path) as src:
    print("CRS del TIFF:", src.crs)
    bounds = src.bounds
    print("Límites del TIFF:")
    print(f"Longitud mínima: {bounds.left}, Longitud máxima: {bounds.right}")
    print(f"Latitud mínima: {bounds.bottom}, Latitud máxima: {bounds.top}")

# Verificar límites del GeoJSON
print("Límites del GeoJSON:")
print(f"Longitud mínima: {geojson_data.bounds.minx.min()}, Longitud máxima: {geojson_data.bounds.maxx.max()}")
print(f"Latitud mínima: {geojson_data.bounds.miny.min()}, Latitud máxima: {geojson_data.bounds.maxy.max()}")

# Transformar CRS si no coinciden
if geojson_data.crs != src.crs:
    print("Transformando CRS del GeoJSON para que coincida con el TIFF...")
    geojson_data = geojson_data.to_crs(src.crs)

# Verificar coordenadas de los puntos
print("Coordenadas de los puntos en el GeoJSON:")
#for _, row in geojson_data.iterrows():
    #print(f"Punto: ({row.geometry.x}, {row.geometry.y})")
