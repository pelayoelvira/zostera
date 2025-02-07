import os
import numpy as np
from tifffile import imread, imwrite

def filter_black_patches(input_images_dir, input_masks_dir, output_images_dir, output_masks_dir, threshold=0.05):
    """
    Filtra parches (imágenes y máscaras) basándose en un umbral de píxeles negros en las máscaras.

    Parámetros:
        input_images_dir (str): Directorio de entrada para las imágenes.
        input_masks_dir (str): Directorio de entrada para las máscaras.
        output_images_dir (str): Directorio de salida para las imágenes filtradas.
        output_masks_dir (str): Directorio de salida para las máscaras filtradas.
        threshold (float): Umbral de píxeles no negros en las máscaras (0-1).

    Salida:
        Las imágenes y máscaras filtradas se guardarán en los directorios de salida.
    """
    # Crear los directorios de salida si no existen
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Listar archivos en los directorios de entrada
    image_files = sorted(os.listdir(input_images_dir))
    mask_files = sorted(os.listdir(input_masks_dir))

    # Procesar y filtrar los archivos
    for image_file, mask_file in zip(image_files, mask_files):
        # Leer la imagen y la máscara
        image_path = os.path.join(input_images_dir, image_file)
        mask_path = os.path.join(input_masks_dir, mask_file)
        
        image = imread(image_path)
        mask = imread(mask_path)
        
        # Filtrar con base en el umbral
        if np.mean(mask) > threshold:
            # Guardar imágenes y máscaras filtradas
            imwrite(os.path.join(output_images_dir, image_file), image)
            imwrite(os.path.join(output_masks_dir, mask_file), mask)

if __name__ == "__main__":
    # Directorios de entrada
    input_images_dir = "Data/filtered_patches/images"
    input_masks_dir = "Data/filtered_patches/masks"

    # Directorios de salida
    output_images_dir = "Data/filtered_patches/filtered_images"
    output_masks_dir = "Data/filtered_patches/filtered_masks"

    # Ejecutar el filtro
    filter_black_patches(input_images_dir, input_masks_dir, output_images_dir, output_masks_dir, threshold=0.05)
