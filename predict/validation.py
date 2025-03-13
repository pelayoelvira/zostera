import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
import numpy as np
from shapely.geometry import box
import seaborn as sns

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
    
    # Inicializar contadores para aciertos y matriz de confusión
    correct = 0
    total = 0
    tp = 0  # Verdaderos positivos
    tn = 0  # Verdaderos negativos
    fp = 0  # Falsos positivos
    fn = 0  # Falsos negativos
    
    for _, row in filtered_points.iterrows():
        longitude, latitude = row['dwc:decimalLongitude'], row['dwc:decimalLatitude']
        habitat = row['dwc:habitat'].strip().lower()
        is_present = habitat in ['nano']
        
        # Convertir las coordenadas en índices de píxel para la máscara
        r, col = rowcol(transform, longitude, latitude)
        # Convertir las coordenadas en índices de píxel para la imagen RGB
        rgb_row, rgb_col = rowcol(rgb_transform, longitude, latitude)
        
        # Verificar en la imagen RGB para descartar puntos en zonas de saturación (blanca)
        if 0 <= rgb_row < rgb_height and 0 <= rgb_col < rgb_width:
            pixel_rgb = rgb_data[:, rgb_row, rgb_col]
            max_val = 255 if np.issubdtype(rgb_dtype, np.uint8) else 65535
            if np.all(pixel_rgb >= max_val - 5):
                continue  # Descarta este punto si cae en zona saturada
        
        # Verificar que el punto se encuentre dentro de la región de la máscara
        if 0 <= r < height and 0 <= col < width:
            mask_present = tiff_data[r, col] > 0
            if is_present:
                if mask_present:
                    tp += 1
                    correct += 1
                else:
                    fn += 1
            else:
                if not mask_present:
                    tn += 1
                    correct += 1
                else:
                    fp += 1
            total += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f'Accuracy: {accuracy:.2f}%')
    print("Confusion Matrix:")
    print(f"TP (True Positives): {tp}")
    print(f"TN (True Negatives): {tn}")
    print(f"FP (False Positives): {fp}")
    print(f"FN (False Negatives): {fn}")

    # Create and save a confusion matrix figure using seaborn
    import matplotlib.pyplot as plt

    conf_matrix = np.array([[tp, fp],
                            [fn, tn]])

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cbar=False,
                     xticklabels=["Predicted Positive", "Predicted Negative"],
                     yticklabels=["Real Positive", "Real Negative"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Real")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.svg", format="svg")
    plt.close()
    return accuracy

# Ejemplo de uso
geojson_path = 'Data/groundtruth_Villaviciosa.geojson'
real_mask = 'Data/RESIZED/image_to_predict/RESIZED_MASK_20240411_VILLAVICIOSA_BORNIZAL3.tif'
mask = 'experiment_1/sin_filtrar.tif'
rgb_path = 'Data/RESIZED/image_to_predict/RESIZED_20240411_VILLAVICIOSA_BORNIZAL3.tif'

accuracy = calculate_accuracy(geojson_path, mask, rgb_path)
