import glob
import os
import numpy as np
import tifffile as tiff


def check_permissions(directory):
    """
    Verifica si el directorio tiene permisos de escritura.

    Args:
        directory (str): Directorio a verificar.

    Returns:
        bool: True si tiene permisos de escritura, False en caso contrario.
    """
    if not os.access(directory, os.W_OK):
        print(f"No tienes permisos de escritura en el directorio: {directory}")
        return False
    return True


def filter_and_save_non_white_patches(image_dir, mask_dir, output_image_dir, output_mask_dir, threshold=0.9):
    """
    Filtra parches de imágenes que no son completamente blancos y guarda las imágenes válidas y sus máscaras.

    Args:
        image_dir (str): Directorio con las imágenes (usar comodines como "*.tif").
        mask_dir (str): Directorio con las máscaras (usar comodines como "*.tif").
        output_image_dir (str): Directorio de salida para las imágenes válidas.
        output_mask_dir (str): Directorio de salida para las máscaras válidas.
        threshold (float): Umbral de blancura para considerar una imagen como completamente blanca. Por defecto, 0.9.

    Returns:
        int: Número de imágenes válidas que pasaron el filtro.
    """
    # Verificar permisos en los directorios de salida
    if not check_permissions(output_image_dir) or not check_permissions(output_mask_dir):
        print("Error de permisos en los directorios de salida.")
        return 0

    # Crear directorios de salida si no existen
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Cargar imágenes y máscaras
    image_names = glob.glob(os.path.join(image_dir, "*.tif"))
    image_names.sort()
    mask_names = glob.glob(os.path.join(mask_dir, "*.tif"))
    mask_names.sort()

    # Inicializar contador para imágenes válidas
    valid_image_count = 0

    for img_name, mask_name in zip(image_names, mask_names):
        # Verificar que no estamos procesando directorios
        if os.path.isdir(img_name) or os.path.isdir(mask_name):
            continue  # Saltar directorios

        # Leer la imagen y la máscara
        img = tiff.imread(img_name)
        mask = tiff.imread(mask_name)
        
        # Detectar el tipo de datos de la imagen y normalizar
        if img.dtype == np.uint16:
            img_normalized = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img_normalized = img.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Tipo de datos desconocido: {img.dtype}")

        # Filtrar imágenes no completamente blancas
        if np.mean(img_normalized) < threshold:
            # Guardar imágenes y máscaras en los nuevos directorios
            valid_image_name = os.path.join(output_image_dir, os.path.basename(img_name))
            valid_mask_name = os.path.join(output_mask_dir, os.path.basename(mask_name))

            # Intentar guardar las imágenes y las máscaras, verificando si los archivos están siendo utilizados
            try:
                tiff.imwrite(valid_image_name, img)  # Guardar en su formato original
                tiff.imwrite(valid_mask_name, mask)
            except PermissionError:
                print(f"Error de permisos al guardar: {valid_image_name} o {valid_mask_name}")
                continue  # Saltar este archivo si no se puede guardar

            valid_image_count += 1

    print(f"Tamaño del dataset resultante: {valid_image_count} imágenes y máscaras válidas")
    return valid_image_count

