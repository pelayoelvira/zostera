import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
import numpy as np
from shapely.geometry import box

def calculate_accuracy(geojson_path, tiff_path, rgb_path):
    # Cargar el groundtruth
    geojson_data = gpd.read_file(geojson_path)
    
    with rasterio.open(tiff_path) as src:
        tiff_data = src.read(1)
        transform = src.transform
        height, width = tiff_data.shape
        # Crear un polígono a partir de los límites del TIFF
        tiff_bounds = box(*src.bounds)
        tiff_crs = src.crs
    
    # Asegurarse de que el groundtruth esté en el mismo CRS que el TIFF
    if geojson_data.crs != tiff_crs:
        geojson_data = geojson_data.to_crs(tiff_crs)
    
    with rasterio.open(rgb_path) as src_rgb:
        rgb_data = src_rgb.read()
        rgb_transform = src_rgb.transform
        rgb_dtype = src_rgb.dtypes[0]
        rgb_height, rgb_width = rgb_data.shape[1:]
    
    # Crear un GeoDataFrame con los límites del TIFF
    tiff_bounds_gdf = gpd.GeoDataFrame({'geometry': [tiff_bounds]}, crs=tiff_crs)
    
    # Clipping: filtrar el groundtruth para que contenga solo los puntos dentro de la región de la máscara
    filtered_points = gpd.clip(geojson_data, tiff_bounds_gdf)
    
    if filtered_points.empty:
        print("No hay puntos dentro de la región de la máscara.")
        return 0
    
    correct = 0
    total = 0
    
    for _, row in filtered_points.iterrows():
        longitude, latitude = row['dwc:decimalLongitude'], row['dwc:decimalLatitude']
        habitat = row['dwc:habitat'].strip().lower()
        is_present = habitat in ['nano', 'nanozos', 'zos']
        
        # Convertir las coordenadas en índices de píxel para la máscara
        col, r = rowcol(transform, longitude, latitude)
        # Convertir las coordenadas en índices de píxel para la imagen RGB
        rgb_col, rgb_row = rowcol(rgb_transform, longitude, latitude)
        
        # Verificar en la imagen RGB para descartar puntos en zonas de relleno (saturación)
        if 0 <= rgb_row < rgb_height and 0 <= rgb_col < rgb_width:
            pixel_rgb = rgb_data[:, rgb_row, rgb_col]
            max_val = 255 if np.issubdtype(rgb_dtype, np.uint8) else 65535
            if np.all(pixel_rgb >= max_val - 5):
                continue  # Descarta este punto si cae en zona saturada (blanca)
        
        # Verificar que el punto se encuentre dentro de la región de la máscara
        if 0 <= r < height and 0 <= col < width:
            pixel_value = tiff_data[r, col] > 0
            if (is_present and pixel_value) or (not is_present and not pixel_value):
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Ejemplo de uso
geojson_path = 'Data/groundtruth_Villaviciosa.geojson'
tiff_path = 'experiment_1/filtrado.tif'
rgb_path = 'Data/RESIZED/image_to_predict_2/RESIZED_20240410_VILLAVICIOSA_IZQ1.tif'

accuracy = calculate_accuracy(geojson_path, tiff_path, rgb_path)
print(f'Porcentaje de aciertos: {accuracy:.2f}%')
