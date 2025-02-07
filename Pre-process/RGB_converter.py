import os
import rasterio


def filter_rgb_channels(input_folder, output_folder):
    """
    Filtra las imágenes TIFF para mantener solo los 3 canales RGB.

    Args:
        input_folder (str): Directorio que contiene las imágenes TIFF de entrada.
        output_folder (str): Directorio donde se guardarán las imágenes con solo 3 canales RGB.

    Returns:
        None
    """
    # Verificar si el directorio de salida existe, si no, crearlo
    os.makedirs(output_folder, exist_ok=True)

    # Iterar sobre los archivos TIFF en la carpeta
    for filename in os.listdir(input_folder):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            input_path = os.path.join(input_folder, filename)

            # Cargar la imagen TIFF usando rasterio
            with rasterio.open(input_path) as src:
                image = src.read()

            # Verificar si la imagen tiene más de 3 canales
            if image.ndim == 3 and image.shape[0] > 3:
                print(f'La imagen {filename} tiene {image.shape[0]} canales. Se reducirá a 3 canales RGB.')
                
                # Seleccionar los primeros 3 canales (R, G, B)
                rgb_image = image[:3, :, :]

                # Guardar la imagen RGB
                output_path = os.path.join(output_folder, filename)
                with rasterio.open(
                    output_path, 'w', 
                    driver='GTiff',
                    height=rgb_image.shape[1], 
                    width=rgb_image.shape[2],
                    count=3, 
                    dtype=rgb_image.dtype,
                    crs=src.crs, 
                    transform=src.transform
                ) as dst:
                    dst.write(rgb_image)

                print(f'Se guardó la imagen RGB: {output_path}')
            else:
                print(f'La imagen {filename} ya tiene 3 canales o menos. No se realizaron cambios.')

