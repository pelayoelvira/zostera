import glob
import numpy as np
import tifffile as tiff

# Función para calcular el tamaño de los datos en MiB
def calculate_dataset_size(image_shape, mask_shape, image_dtype=np.float32, mask_dtype=np.float32):
    image_dtype_size = np.dtype(image_dtype).itemsize  # float32 = 4 bytes por pixel
    mask_dtype_size = np.dtype(mask_dtype).itemsize  # float32 = 4 bytes por pixel

    # Calcular tamaño de una imagen y una máscara
    image_size_bytes = np.prod(image_shape) * image_dtype_size
    mask_size_bytes = np.prod(mask_shape) * mask_dtype_size

    image_size_mib = image_size_bytes / (1024 ** 2)
    mask_size_mib = mask_size_bytes / (1024 ** 2)

    return image_size_mib, mask_size_mib

# Dimensiones y tipo de dato de las imágenes
img_height = 512  # ejemplo: alto de las imágenes
img_width = 512  # ejemplo: ancho de las imágenes
img_channels = 3  # ejemplo: canales RGB
image_shape = (img_height, img_width, img_channels)

# Dimensiones y tipo de dato de las máscaras
mask_shape = (img_height, img_width, 1)  # máscaras unidimensionales (1 canal)
mask_dtype = np.float32  # cambiar a uint8 si es binaria

# Cargar el dataset
image_names = glob.glob("Data/filtered_patches/images_512/*.tif")
mask_names = glob.glob("Data/filtered_patches/masks_512/*.tif")

# Calcular tamaño por imagen y máscara
image_size_mib, mask_size_mib = calculate_dataset_size(image_shape, mask_shape)

# Número total de imágenes y máscaras
num_images = len(image_names)
num_masks = len(mask_names)

# Calcular tamaño total del dataset
total_image_size_mib = image_size_mib * num_images
total_mask_size_mib = mask_size_mib * num_masks
total_dataset_size_mib = total_image_size_mib + total_mask_size_mib

# Mostrar resultados
print(f"Tamaño total de las imágenes: {total_image_size_mib:.2f} MiB")
print(f"Tamaño total de las máscaras: {total_mask_size_mib:.2f} MiB")
print(f"Tamaño total del dataset (imágenes + máscaras): {total_dataset_size_mib:.2f} MiB")
