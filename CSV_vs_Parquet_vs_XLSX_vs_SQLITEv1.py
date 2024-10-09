import os
import pandas as pd
import numpy as np

# Paramètres pour générer un DataFrame de 100 Mo
rows = 100_000  # Nombre de lignes
cols = 100        # Nombre de colonnes

# Génération d'un DataFrame avec des données aléatoires
df = pd.DataFrame(np.random.randn(rows, cols), columns=[f'col_{i}' for i in range(cols)])

# Vérifier la taille en mémoire du DataFrame
memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # en Mo
print(f"Taille du DataFrame : {memory_usage:.2f} Mo")

# Sauvegarder en CSV
csv_file = "dataframe_100MB.csv"
df.to_csv(csv_file, index=False)

# Sauvegarder en Parquet
parquet_file = "dataframe_100MB.parquet"
df.to_parquet(parquet_file)

# Sauvegarder en Excel
excel_file = "dataframe_100MB.xlsx"
df.to_excel(excel_file, index=False)



# Fichiers créés
files = [csv_file, parquet_file, excel_file]

# Fonction pour calculer la taille des fichiers en Mo
def file_size_in_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)  # en Mo

# Calcul des tailles des fichiers
file_sizes = {file: file_size_in_mb(file) for file in files}

# Affichage des tailles de fichiers
for file, size in file_sizes.items():
    print(f"Taille de {file} : {size:.2f} Mo")
