import numpy as np
import os
import tifffile as tiff
from patchify import patchify
import rasterio
import gc

def crop_to_patch_size(image, patch_size):
    '''Recorta la imagen para que sus dimensiones sean divisibles por el tamaño del parche.'''
    new_height = (image.shape[0] // patch_size) * patch_size
    new_width = (image.shape[1] // patch_size) * patch_size
    if len(image.shape) == 3:  # Imagen a color
        return image[:new_height, :new_width, :]
    else:  # Máscara binaria
        return image[:new_height, :new_width]

def generate_patches(images_dir, masks_dir, image_output_dir, mask_output_dir, patch_size=512):
    """
    Genera parches a partir de imágenes y máscaras, procesando y recortando cada par según el tamaño especificado.
    Parámetros:
        images_dir (str): Directorio que contiene las imágenes de entrada.
        masks_dir (str): Directorio que contiene las máscaras correspondientes a las imágenes.
        image_output_dir (str): Directorio donde se guardarán los parches generados de las imágenes.
        mask_output_dir (str): Directorio donde se guardarán los parches generados de las máscaras.
        patch_size (int, opcional): Tamaño (alto y ancho) de cada parche. Por defecto es 512.
    Operaciones:
        - Crea los directorios de salida si no existen.
        - Procesa cada imagen que cumpla con el patrón "RESIZED_*.tif" y verifica la existencia de la máscara asociada.
        - Abre y transpone la imagen para ajustarla al formato (alto, ancho, canales).
        - Recorta la imagen y la máscara según el tamaño del parche.
        - Genera parches de imagen y máscara utilizando la función patchify.
        - Guarda cada parche en los directorios de salida indicados.
        - Libera memoria utilizando gc.collect.
    Excepciones:
        - Lanza FileNotFoundError si no se encuentra la máscara correspondiente a una imagen.
    """
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Procesar cada imagen en el directorio
    for image_file in os.listdir(images_dir):
        if image_file.startswith("RESIZED_") and image_file.endswith(".tif"):
            mask_file = image_file.replace("RESIZED_", "RESIZED_MASK_")
            
            # Verificar existencia de la máscara antes de proceder
            if not os.path.exists(os.path.join(masks_dir, mask_file)):
                raise FileNotFoundError(f"Máscara no encontrada para {image_file}.")
            
            with rasterio.open(os.path.join(images_dir, image_file)) as src:
                large_image_stack = src.read()
            large_mask_stack = tiff.imread(os.path.join(masks_dir, mask_file))

            print(f"Procesando {image_file} con máscara {mask_file}")
            print(f"Dimensiones de la imagen: {large_image_stack.shape}")
            print(f"Dimensiones de la máscara: {large_mask_stack.shape}")
            
            # Transponer la imagen para adecuarla al formato esperado (H, W, C)
            large_image_stack = np.transpose(large_image_stack, (1, 2, 0))

            # Recortar la imagen y la máscara
            large_image_stack = crop_to_patch_size(large_image_stack, patch_size)
            large_mask_stack = crop_to_patch_size(large_mask_stack, patch_size)

            # Crear los parches
            patches_img = patchify(large_image_stack, (patch_size, patch_size, 3), step=patch_size)
            patches_mask = patchify(large_mask_stack, (patch_size, patch_size), step=patch_size)

            print(f"Dimensiones de los parches de la imagen: {patches_img.shape}")
            print(f"Dimensiones de los parches de la máscara: {patches_mask.shape}")

            # Guardar cada parche en los directorios correspondientes
            k = 0
            base_name = image_file.replace("RESIZED_", "").replace(".tif", "")
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, :, :, :]
                    image_patch_filename = os.path.join(image_output_dir, f'patch_image_{base_name}_{k}.tif')
                    tiff.imwrite(image_patch_filename, single_patch_img.squeeze())
                    
                    single_patch_mask = patches_mask[i, j, :, :]
                    mask_patch_filename = os.path.join(mask_output_dir, f'patch_mask_{base_name}_{k}.tif')
                    tiff.imwrite(mask_patch_filename, single_patch_mask)
                    
                    k += 1

            print(f"Se generaron {k} parches para {image_file}.")

    patch_files = [f for f in os.listdir(image_output_dir)]
    print(f"Se han generado y guardado {len(patch_files)} parches en {image_output_dir}.")
    
    # Liberar memoria si es necesario
    gc.collect()

    return
