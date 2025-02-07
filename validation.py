import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
import numpy as np

def calculate_accuracy(geojson_path, tiff_path, rgb_path, locality_name):
    geojson_data = gpd.read_file(geojson_path)

    # Filtrar por localidad
    filtered_points = geojson_data[geojson_data['dwc:locality'] == locality_name]

    if filtered_points.empty:
        print(f"No se encontraron puntos para la localidad: {locality_name}")
        return 0

    with rasterio.open(tiff_path) as src:
        tiff_data = src.read(1)
        transform = src.transform
        height, width = tiff_data.shape
        tiff_bounds = src.bounds  # Obtener límites del TIFF

    with rasterio.open(rgb_path) as src_rgb:
        rgb_data = src_rgb.read()
        rgb_transform = src_rgb.transform
        rgb_dtype = src_rgb.dtypes[0]  # Verificar el tipo de datos
        print("Tipo de datos del TIFF RGB:", rgb_dtype)
        rgb_height, rgb_width = rgb_data.shape[1:]
        rgb_bounds = src_rgb.bounds

    # Filtrar puntos dentro de los límites del TIFF
    def is_within_bounds(row):
        longitude, latitude = row['dwc:decimalLongitude'], row['dwc:decimalLatitude']
        return (tiff_bounds.left <= longitude <= tiff_bounds.right) and (tiff_bounds.bottom <= latitude <= tiff_bounds.top)

    filtered_points = filtered_points[filtered_points.apply(is_within_bounds, axis=1)]

    if filtered_points.empty:
        print(f"Todos los puntos de {locality_name} están fuera de los límites del TIFF.")
        return 0

    correct = 0
    total = 0

    for _, row in filtered_points.iterrows():
        latitude = row['dwc:decimalLatitude']
        longitude = row['dwc:decimalLongitude']
        habitat = row['dwc:habitat']

        col, row = rowcol(transform, longitude, latitude)
        rgb_col, rgb_row = rowcol(rgb_transform, longitude, latitude)

        if 0 <= rgb_row < rgb_height and 0 <= rgb_col < rgb_width:
            pixel_rgb = rgb_data[:, rgb_row, rgb_col]
            max_val = 255 if np.issubdtype(rgb_dtype, np.uint8) else 65535
            if np.all(pixel_rgb >= max_val - 5):  # Comprobar si la región es blanca
                continue  # Omitir este punto

        if 0 <= row < height and 0 <= col < width:
            pixel_value = tiff_data[row, col]
            pixel_value = 1 if pixel_value > 0 else 0
            
            is_present = habitat.strip().lower() in ['nano', 'nanozos', 'zos']
            if (is_present and pixel_value == 1) or (not is_present and pixel_value == 0):
                correct += 1
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Ejemplo de uso
geojson_path = 'Data/1_groundtruth.geojson'
tiff_path = 'Data/RESIZED/NEW_MASKS/RESIZED_MASK_20240212_VILLAVICIOSA_ENCIENONA.tif'
rgb_path = 'Data/RESIZED/IMAGES/RESIZED_20240212_VILLAVICIOSA_ENCIENONA.tif'
locality_name = 'VILLAVICIOSA_ENCIENONA'

accuracy = calculate_accuracy(geojson_path, tiff_path, rgb_path, locality_name)
print(f'Porcentaje de aciertos para {locality_name}: {accuracy:.2f}%')
