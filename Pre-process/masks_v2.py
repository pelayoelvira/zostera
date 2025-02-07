import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

def poly_from_utm(geom, transform):
    """
    Reproyecta una geometría (Polygon o MultiPolygon) manteniendo sus huecos.
    
    Parámetros:
      geom: geometría de tipo Polygon o MultiPolygon.
      transform: transformación de la imagen ráster.
      
    Retorna:
      Una lista de polígonos reproyectados (cada uno con sus huecos, si los tiene).
    """
    # Si la geometría es un polígono simple, se reproyecta su contorno exterior y sus huecos.
    if geom.geom_type == 'Polygon':
        # Reproyectar el contorno exterior
        poly_pts = [~transform * tuple(coord) for coord in geom.exterior.coords]
        # Reproyectar cada hueco (si existen)
        holes = [[~transform * tuple(coord) for coord in interior.coords] for interior in geom.interiors]
        return [Polygon(poly_pts, holes=holes)]
    
    # Si la geometría es un MultiPolygon, se procesa cada polígono individualmente
    elif geom.geom_type == 'MultiPolygon':
        polys = []
        for polygon in geom.geoms:
            polys.extend(poly_from_utm(polygon, transform))
        return polys
    else:
        raise ValueError(f"Tipo de geometría no soportado: {geom.geom_type}")

# Rutas de entrada y salida
input_dir = r'Data\0_orthomosaics'     # Carpeta con los archivos ráster
mask_dir = r'Data\MASKS__'             # Carpeta para guardar las máscaras
shape_path = r'Data\Nanozostera_noltei.geojson'  # Archivo GeoJSON

# Crear la carpeta de máscaras si no existe
os.makedirs(mask_dir, exist_ok=True)

# Leer el GeoJSON y filtrar por localidad
geojson = gpd.read_file(shape_path)
geojson = geojson[geojson['localidad'] == 'Villaviciosa']

# Iterar sobre los archivos .tif en la carpeta de ráster
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        raster_path = os.path.join(input_dir, filename)
        
        with rasterio.open(raster_path, "r") as src:
            # Si el CRS del GeoJSON no coincide con el de la imagen, se reproyecta
            if geojson.crs != src.crs:
                geojson = geojson.to_crs(src.crs)
            
            poly_shp = []  # Lista para almacenar los polígonos transformados
            im_size = (src.meta['height'], src.meta['width'])
            
            # Iterar sobre los registros del GeoJSON
            for _, row in geojson.iterrows():
                # Procesar solo geometrías tipo Polygon o MultiPolygon
                if row['geometry'].geom_type in ['Polygon', 'MultiPolygon']:
                    # poly_from_utm retorna una lista de polígonos
                    polygons = poly_from_utm(row['geometry'], src.meta['transform'])
                    poly_shp.extend(polygons)
            
            # Rasterizar: se asigna el valor 1 a los polígonos (áreas de interés)
            # y se mantiene 0 en el fondo (incluyendo los huecos)
            mask = rasterize(
                [(poly, 1) for poly in poly_shp],
                out_shape=im_size,
                fill=0,
                all_touched=True
            )
            mask = mask.astype("uint8")
            
            # Guardar la máscara resultante, multiplicando por 255 para obtener blanco (255) y negro (0)
            mask_filename = f"MASK_{filename}"
            save_path = os.path.join(mask_dir, mask_filename)
            bin_mask_meta = src.meta.copy()
            bin_mask_meta.update({'count': 1, 'dtype': 'uint8'})
            
            with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
                dst.write(mask * 255, 1)
