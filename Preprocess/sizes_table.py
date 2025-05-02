import os
import tifffile

# Ruta al directorio
directory = r'Data/RESIZED/IMAGES_2'

# Recoger todos los .tif
files = sorted(f for f in os.listdir(directory) if f.lower().endswith('.tif'))

# Procesar cada imagen
rows = []
for fn in files:
    path = os.path.join(directory, fn)
    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape  # dimensiones (h, w) o (h, w, c)
        if len(shape) == 2:
            h, w = shape
        else:
            h, w = shape[:2]
    size_mb = os.path.getsize(path) / (1024 * 1024)
    rows.append((fn, f"{w}×{h}", f"{size_mb:.2f}"))

# Generar salida LaTeX
print(r"\begin{table}[ht]")
print(r"  \centering")
print(r"  \caption{Resumen de ortomosaicos}")
print(r"  \label{tab:ortomosaicos}")
print(r"  \begin{tabular}{lcc}")
print(r"    \toprule")
print(r"    Nombre de la imagen & Dimensiones (px) & Tamaño (MB) \\")
print(r"    \midrule")
for name, dims, size in rows:
    print(f"    {name} & {dims} & {size} \\\\")
print(r"    \bottomrule")
print(r"  \end{tabular}")
print(r"\end{table}")
