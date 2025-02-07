import os
import tifffile as tiff
import cv2
import numpy as np

# Directorios que contienen las imágenes y las máscaras redimensionadas
resized_image_dir = r'Data\nueva_imagen'
resized_mask_dir = r'Data\nueva_mask'

# Iterar sobre las imágenes redimensionadas
for image_file in os.listdir(resized_image_dir):
    if not image_file.endswith(".tif"):
        continue  # Ignorar archivos que no sean TIFF
    
    # Construir el nombre de la máscara
    mask_file = f"RESIZED_MASK{image_file[7:]}"  # Eliminar "RESIZED_" del nombre

    image_path = os.path.join(resized_image_dir, image_file)
    mask_path = os.path.join(resized_mask_dir, mask_file)
    
    # Cargar la imagen y la máscara
    large_image = tiff.imread(image_path) # Convertir a uint16
    large_mask = tiff.imread(mask_path)  # Mantener como uint8

    # Imprimir las dimensiones y el tipo de datos
    print('El formato de tamaño de la imagen es alto x ancho x canales:')
    print(f"Dimensiones de la imagen redimensionada: {large_image.shape}, Tipo: {large_image.dtype}")
    print(large_image[0, 0, :])  # Imprimir los valores de los 3 canales en el píxel superior izquierdo
    print(f"Dimensiones de la máscara redimensionada: {large_mask.shape}, Tipo: {large_mask.dtype}")
