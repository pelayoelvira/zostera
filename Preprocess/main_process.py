from Preprocess.geo_resized import resize_rasters
from Preprocess.RGB_converter import filter_rgb_channels
from Preprocess.postprocess import postprocess_masks
from Preprocess.create_patches import generate_patches
from Preprocess.filter_patches import filter_and_save_non_white_patches
import os
import shutil  

# Directorios de entrada y salida para las imágenes y las máscaras
input_image_dir = r'Data/0_orthomosaics'
input_mask_dir = r'Data/MASKS'

resized_image_dir = r'Data/RESIZED/IMAGES_og'
resized_mask_dir = r'Data/RESIZED/MASKS_og'
resized_new_mask_dir = r'Data/RESIZED/NEW_MASKS_og'

patch_output_image_dir = r'Data/patches/images_og'
patch_output_mask_dir = r'Data/patches/masks_og'

filtered_image_dir = r'Data/filtered_patches/images_og'
filtered_mask_dir = r'Data/filtered_patches/masks_og'

# Tamaño del parche
patch_size = 512

# Directorio donde se moverán las imágenes y máscaras seleccionadas
predict_image_dir = r'Data/RESIZED/image_to_predict_og'

# Archivos específicos a mover para validacion a modo de produccion
image_to_move = "RESIZED_20240411_VILLAVICIOSA_BORNIZAL3.tif"
mask_to_move = "RESIZED_MASK_20240411_VILLAVICIOSA_BORNIZAL3.tif"

os.makedirs(predict_image_dir, exist_ok=True)

if __name__ == "__main__":
    # Paso 0: Crear directorios de salida si no existen
    os.makedirs(patch_output_image_dir, exist_ok=True)
    os.makedirs(patch_output_mask_dir, exist_ok=True)
    os.makedirs(filtered_image_dir, exist_ok=True)
    os.makedirs(filtered_mask_dir, exist_ok=True)
    
    print("Creando directorios de salida...")
    
    # Paso 1: Redimensionar imágenes y máscaras
    print("Iniciando redimensionado de imágenes y máscaras...")
    resize_rasters(input_image_dir, input_mask_dir, resized_image_dir, resized_mask_dir, dst_res=None) 
    print("Redimensionado completado.")

    # Paso 2: Filtrar canales RGB
    print("Iniciando filtrado de canales RGB...")
    filter_rgb_channels(resized_image_dir, resized_image_dir)  # Filtrado de los canales RGB en las imágenes redimensionadas
    print("Filtrado de canales RGB completado.")

    # Paso 3: Postprocesar máscaras
    print("Iniciando postprocesado de máscaras...")
    postprocess_masks(input_image_dir, input_mask_dir, resized_new_mask_dir)  # Postprocesar las máscaras
    print("Postprocesado de máscaras completado.")

    # Mover la imagen y su máscara específica
    print("Moviendo imagen y máscara  a 'image_to_predict'...")
    image_path = os.path.join(input_image_dir, image_to_move)
    mask_path = os.path.join(resized_new_mask_dir, mask_to_move)
    try:
        shutil.move(image_path, os.path.join(predict_image_dir, image_to_move))
        shutil.move(mask_path, os.path.join(predict_image_dir, mask_to_move))
        print(f"Imagen '{image_to_move}' y máscara '{mask_to_move}' movidas exitosamente.")
    except FileNotFoundError as e:
        print(f"Error al mover archivos: {e}")
    
    # Paso 4: Generación de parches de las imágenes y máscaras redimensionadas
    print("Iniciando generación de parches...")
    generate_patches(input_image_dir, resized_new_mask_dir, patch_output_image_dir, patch_output_mask_dir, patch_size)
    print("Generación de parches completada.")

    # Paso 5: Filtrar parches no blancos
    print("Iniciando filtrado de parches blancos...")
    filter_and_save_non_white_patches(patch_output_image_dir, patch_output_mask_dir, filtered_image_dir, filtered_mask_dir, threshold=0.9)
    print("Filtrado de parches completado.")

    print("Flujo completo ejecutado exitosamente.")
    
    
    # 1. dst_res=0.0000012) #0.1 metros
    # 2. dst_res=0.0000006 #0.05 metros
