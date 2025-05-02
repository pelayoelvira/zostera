import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib.font_manager as fm

# Establecer el tema de Seaborn
sns.set_theme(style="whitegrid", palette="pastel",context='paper', font_scale=1.3)  


# Ruta al archivo GeoJSON de groundtruth
geojson_path = "Data/groundtruth_Villaviciosa.geojson"

# Cargar el GeoJSON en un GeoDataFrame
gdf = gpd.read_file(geojson_path)

# Número total de puntos en el archivo
total_points = len(gdf)

# Se consideran como "positivos" (presencia de N. noltei) aquellos puntos cuyo valor en 'dwc:habitat'
# es "nano", "nanozos" o "zos" (ignorando mayúsculas/minúsculas). El resto se considera "negativo".
positive_mask = gdf["dwc:habitat"].str.lower().isin(["nano"])
positive_count = positive_mask.sum()
negative_count = total_points - positive_count

# Imprimir el resumen de la distribución
print(f"Total de puntos: {total_points}")
print(f"Puntos positivos (presencia de N. noltei): {positive_count}")
print(f"Puntos negativos (ausencia de N. noltei): {negative_count}")

# Generar un histograma (gráfico de barras) para visualizar la distribución
labels = ["Positivos", "Negativos"]
counts = [positive_count, negative_count]
plt.figure(figsize=(8,6))
ax = sns.barplot(x=labels, y=counts, palette=["#4E79A7", "#F28E2B"])  # Colores claros y formales

# Calcular el total para obtener el porcentaje en cada categoría
total = sum(counts)
for p in ax.patches:
    height = p.get_height()
    percentage = height / total * 100
    ax.annotate(f'{percentage:.1f}%', 
                (p.get_x() + p.get_width() / 2, height), 
                ha='center', va='bottom')

plt.xlabel("Categoría")
plt.ylabel("Número de puntos")
plt.tight_layout()

# Guardar el gráfico como archivo SVG
plt.savefig("groundtruth_distribution.svg", format="svg", bbox_inches="tight")
plt.show()
