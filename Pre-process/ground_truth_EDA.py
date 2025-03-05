import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

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
sns.barplot(x=labels, y=counts)
plt.xlabel("Categoría")
plt.ylabel("Número de puntos")
plt.title("Distribución de puntos en groundtruth")

# Guardar el gráfico como archivo SVG
plt.savefig("groundtruth_distribution.svg", format="svg", bbox_inches="tight")
plt.show()
