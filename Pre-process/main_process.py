from geo_resized import resize_rasters
from RGB_converter import filter_rgb_channels
from postprocess import postprocess_masks
from create_patches import generate_patches
from filter_patches import filter_and_save_non_white_patches
import os

# Directorios de entrada y salida para las imágenes y las máscaras
input_image_dir = r'Data\0_orthomosaics'
input_mask_dir = r'Data\MASKS'

resized_image_dir = r'Data\RESIZED\IMAGES'
resized_mask_dir = r'Data\RESIZED\MASKS'
resized_new_mask_dir = r'Data\RESIZED\NEW_MASKS'

patch_output_image_dir = r'Data\patches\images'
patch_output_mask_dir = r'Data\patches\masks'

filtered_image_dir = r'Data\filtered_patches\images'
filtered_mask_dir = r'Data\filtered_patches\masks'

# Tamaño del parche
patch_size = 512

if __name__ == "__main__":
    
     # Paso 0: Crear directorios de salida si no existen
    os.makedirs(patch_output_image_dir, exist_ok=True)
    os.makedirs(patch_output_mask_dir, exist_ok=True)
    os.makedirs(filtered_image_dir, exist_ok=True)
    os.makedirs(filtered_mask_dir, exist_ok=True)
    
    # Paso 1: Redimensionar imágenes y máscaras
    print("Iniciando redimensionado de imágenes y máscaras...")
    resize_rasters(input_image_dir, input_mask_dir, resized_image_dir, resized_mask_dir, dst_res=0.000001) #0.08m cada pixel
    print("Redimensionado completado.")

    # Paso 2: Filtrar canales RGB
    print("Iniciando filtrado de canales RGB...")
    filter_rgb_channels(resized_image_dir, resized_image_dir)  # Filtrado de los canales RGB en las imágenes redimensionadas
    print("Filtrado de canales RGB completado.")

    # Paso 3: Postprocesar máscaras
    print("Iniciando postprocesado de máscaras...")
    postprocess_masks(resized_image_dir, resized_mask_dir, resized_new_mask_dir)  # Postprocesar las máscaras
    print("Postprocesado de máscaras completado.")

    # Paso 4: Generación de parches de las imágenes y máscaras redimensionadas
    print("Iniciando generación de parches...")
    generate_patches(resized_image_dir, resized_new_mask_dir, patch_output_image_dir, patch_output_mask_dir, patch_size)
    print("Generación de parches completada.")

    # Paso 5: Filtrar parches no blancos
    print("Iniciando filtrado de parches blancos...")
    filter_and_save_non_white_patches(patch_output_image_dir, patch_output_mask_dir, filtered_image_dir, filtered_mask_dir, threshold=0.9)
    print("Filtrado de parches completado.")

    print("Flujo completo ejecutado exitosamente.")
