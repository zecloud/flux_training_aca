import os
from PIL import Image
import shutil

# Chemins des dossiers
source_folder = "Rembrandt"  # Remplace par le chemin de ton dossier source
destination_folder = "dataset"  # Remplace par le chemin de ton dossier de destination

# Crée le dossier de destination s'il n'existe pas
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Parcourt les fichiers dans le dossier source
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    # Vérifie si le fichier est une image
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        try:
            # Ouvre l'image pour vérifier sa résolution
            with Image.open(file_path) as img:
                width, height = img.size

                # Vérifie si l'une des dimensions est supérieure à 1024 pixels
                if width > 1024 or height > 1024:
                    # Copie l'image dans le dossier de destination
                    shutil.copy(file_path, destination_folder)
                    print(f"Copié: {filename} ({width}x{height})")
                else:
                    print(f"Ignoré: {filename} ({width}x{height})")

        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

print("Tri terminé.")
