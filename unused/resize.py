import os
import tifffile
import cv2
import numpy as np

# Directorios que contienen las imágenes y las máscaras
image_dir = r'Data\nueva_imagen'
mask_dir = r'Data\nueva_mask'
resized_image_dir = r'Data\nueva_imagen'
resized_mask_dir = r'Data\nueva_mask'

# Asegurarse de que las carpetas de salida existan
os.makedirs(resized_image_dir, exist_ok=True)
os.makedirs(resized_mask_dir, exist_ok=True)

# Iterar sobre las imágenes para redimensionar
for image_file in os.listdir(image_dir):
    if not image_file.endswith(".tif"):
        continue  # Ignorar archivos que no sean TIFF
    
    # Construir el nombre de la máscara
    mask_file = f"MASK_{image_file}"

    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    # Cargar la imagen y la máscara
    image = tifffile.imread(image_path)
    mask = tifffile.imread(mask_path)
    
    # Verificar que la máscara tenga la misma altura y ancho que la imagen
    if image.shape[0:2] != mask.shape[0:2]:
        print(f"Advertencia: Las dimensiones de {image_file} y {mask_file} no coinciden.")
        continue  # Ignorar esta pareja y seguir con el siguiente

    # Determinar nuevas dimensiones
    old_height, old_width = image.shape[:2]
    if abs(old_height - old_width) <= 2000:  # Cuadrada
        new_height, new_width = 6000, 6000
    elif old_height > old_width:  # Más alta que ancha
        new_height, new_width = 7000, 5000
    else:  # Más ancha que alta
        new_height, new_width = 5000, 7000

    # Redimensionar imagen y máscara
    resized_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    for i in range(3):  # Solo los 3 canales RGB
        resized_image[:, :, i] = cv2.resize(image[:, :, i], (new_width, new_height))
    
    resized_mask = cv2.resize(mask, (new_width, new_height))

    # Guardar las imágenes y máscaras redimensionadas
    tifffile.imwrite(os.path.join(resized_image_dir, f"RESIZED_{image_file}"), resized_image)
    tifffile.imwrite(os.path.join(resized_mask_dir, f"RESIZED_{mask_file}"), resized_mask)

    # Mostrar dimensiones
    print(f"Imagen {image_file} redimensionada a: {resized_image.shape}")
    print(f"Máscara {mask_file} redimensionada a: {resized_mask.shape}")
