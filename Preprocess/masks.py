import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np

# Función para reproyectar el polígono según el sistema de coordenadas
def poly_from_utm(polygon, transform):
    '''Función para reproyectar el polígono del sistema de coordenadas del GeoJson al de la imagen raster
    Entradas: polígono, transformación de la imagen raster
    Salida: polígono reproyectado
    '''
    poly_pts = []
    poly = unary_union(polygon) # Unary_union une multiples polígonos en un solo polígono, si es Polygon da el mismo polígono
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i)) #se hace una trasnformacion inversa a cada coodenada y se almacena como tupla
    return Polygon(poly_pts) # se retonra la region en cuestion

input_dir = r'Data\nueva_imagen'  # Carpeta con los archivos raster
mask_dir = r'Data\nueva_mask'            # Carpeta para guardar las máscaras
shape_path = r'Data\Nanozostera_noltei.geojson'  # Archivo GeoJSON

geojson = gpd.read_file(shape_path)

# Se filtran los  registros donde la localidad sea "Villaviciosa"
geojson = geojson[geojson['localidad'] == 'Villaviciosa']

# Iterar sobre los archivos tiff en la carpeta
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):  
        raster_path = os.path.join(input_dir, filename)
        
        with rasterio.open(raster_path, "r") as src:
            #print(src.meta)
            # Si el sistema de coordenadas de la imagen no coincide con el del geosjon, se cambian a las coordenadas de la imagen
            if geojson.crs != src.crs:
                geojson = geojson.to_crs(src.crs)
            
            
            poly_shp = [] # Se crea una lista vacia para almacenar los poligonos
            im_size = (src.meta['height'], src.meta['width']) 
            
            # Con iterrows se itera sobre las filas del GeoDataFrame, que contiene los polígonos
            for num, row in geojson.iterrows():
                if isinstance(row['geometry'], (MultiPolygon, Polygon)): # se verifica si es un poligono o un multipoligono
                    #print(row)
                    #utiliza las coordenadas que se encuentran en la columna geometry y las que se encuentran en la columna transform para
                    #saber que poligono se encuentra en que lugar de la imagen
                    poly = poly_from_utm(row['geometry'], src.meta['transform']) #se llama a la funcion poly_from_utm, pasando el poligono y la transformacion
                    poly_shp.append(poly)
            
            mask = rasterize(shapes=poly_shp, out_shape=im_size) # Convirte los poligonos en una imagen
            mask = mask.astype("uint8")
            
            mask_filename = f"MASK_{filename}"
            save_path = os.path.join(mask_dir, mask_filename)
            bin_mask_meta = src.meta.copy() # Se copian los metadatos de la imagen
            bin_mask_meta.update({'count': 1, 'dtype': 'uint8'})  # Se actualizan los metadatos para la máscara binaria
            
            with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
                dst.write(mask * 255, 1)  