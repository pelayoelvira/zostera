import os
import rasterio
import numpy as np

import os
import rasterio
import numpy as np

def postprocess_masks(resized_image_dir, resized_mask_dir, output_mask_dir):
    """
    Modifica las máscaras binarias basándose en las imágenes RGB para eliminar áreas completamente blancas.

    Args:
        resized_image_dir (str): Directorio que contiene las imágenes redimensionadas.
        resized_mask_dir (str): Directorio que contiene las máscaras redimensionadas.
        output_mask_dir (str): Directorio donde se guardarán las máscaras modificadas (se sobreescriben).

    Returns:
        None
    """
    os.makedirs(output_mask_dir, exist_ok=True)

    # Iterar sobre las imágenes redimensionadas
    for image_file in os.listdir(resized_image_dir):
        if not image_file.endswith(".tif"):
            continue  
        
        # Construir el nombre de la máscara correspondiente
        mask_file = f"RESIZED_MASK{image_file[7:]}"  # Eliminar "RESIZED_" del nombre de la imagen
        
        image_path = os.path.join(resized_image_dir, image_file)
        mask_path = os.path.join(resized_mask_dir, mask_file)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada para {image_file}")

        try:
            # Cargar la imagen y la máscara usando rasterio
            with rasterio.open(image_path) as src_image:
                rgb_image = src_image.read()
                image_meta = src_image.meta

            with rasterio.open(mask_path) as src_mask:
                binary_image = src_mask.read(1) 
                mask_meta = src_mask.meta

            # Verificar que ambas imágenes tengan el mismo tamaño
            if rgb_image.shape[1:] != binary_image.shape:
                raise ValueError(f"Las imágenes no tienen el mismo tamaño: {image_file} y su máscara correspondiente.")

            # Determinar el valor de blanco según el tipo de dato de la imagen RGB
            if rgb_image.dtype == np.uint8:
                white_value = 255
            elif rgb_image.dtype == np.uint16:
                white_value = 65535
            else:
                raise ValueError("Tipo de datos de la imagen RGB no soportado. Use uint8 o uint16.")

            # Crear una máscara donde la imagen RGB es completamente blanca
            white_mask = np.all(rgb_image == white_value, axis=0)

            # Modificar la imagen binaria: donde la máscara es True, se ponen a 0 (negro)
            binary_image[white_mask] = 0

            # Guardar la nueva imagen binaria con el mismo nombre, sobrescribiendo
            new_mask_path = os.path.join(output_mask_dir, mask_file)
            mask_meta.update(dtype=rasterio.uint8, count=1)

            with rasterio.open(new_mask_path, 'w', **mask_meta) as dst:
                dst.write(binary_image, 1)

            print(f"Imagen binaria modificada guardada: {new_mask_path}")

        except Exception as e:
            print(f"Error procesando {image_file} y {mask_file}: {e}")
