import os
import rasterio
from rasterio.enums import Resampling


def resize_rasters(image_dir, mask_dir, resized_image_dir, resized_mask_dir, dst_res):
    """
    Redimensiona imágenes y sus máscaras a una resolución deseada.

    Args:
        image_dir (str): Directorio que contiene las imágenes originales.
        mask_dir (str): Directorio que contiene las máscaras originales.
        resized_image_dir (str): Directorio donde se guardarán las imágenes redimensionadas.
        resized_mask_dir (str): Directorio donde se guardarán las máscaras redimensionadas.
        dst_res (float): Resolución deseada en grados por píxel.

    Returns:
        None
    """
    # Crear carpetas de salida si no existen
    os.makedirs(resized_image_dir, exist_ok=True)
    os.makedirs(resized_mask_dir, exist_ok=True)

    def process_raster(src_path, dst_path, resampling_method):
        """Procesa un raster para redimensionarlo."""
        with rasterio.open(src_path) as src:
            transform = src.transform
            crs = src.crs
            current_res = transform.a
            scale_factor = current_res / dst_res
            new_width = int(src.width * scale_factor)
            print(f"Ancho original: {src.width}, Ancho nuevo: {new_width}")
            new_height = int(src.height * scale_factor)
            print(f"Alto original: {src.height}, Alto nuevo: {new_height}")
            new_transform = transform * transform.scale(
                (src.width / new_width), (src.height / new_height)
            )

            # Leer y redimensionar los datos
            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=resampling_method
            )

            # Guardar el raster redimensionado
            with rasterio.open(
                dst_path, 'w',
                driver='GTiff',
                count=src.count,
                dtype=data.dtype,
                crs=crs,
                transform=new_transform,
                width=new_width,
                height=new_height,
                compress='lzw'
            ) as dst:
                dst.write(data)

    # Iterar sobre las imágenes
    for image_file in os.listdir(image_dir):
        if not image_file.endswith(".tif"):
            continue

        mask_file = f"MASK_{image_file}"
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        resized_image_path = os.path.join(resized_image_dir, f"RESIZED_{image_file}")
        resized_mask_path = os.path.join(resized_mask_dir, f"RESIZED_{mask_file}")

        if not os.path.exists(mask_path):
            print(f"Advertencia: Máscara no encontrada para {image_file}.")
            continue

        try:
            # Procesar la imagen con resampling bilinear
            process_raster(image_path, resized_image_path, Resampling.bilinear)

            # Procesar la máscara con resampling nearest
            process_raster(mask_path, resized_mask_path, Resampling.nearest)

            print(f"Procesados: {image_file} y {mask_file}")
        except Exception as e:
            print(f"Error procesando {image_file}: {e}")
