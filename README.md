# Projet de détection d'objets de circulation "FluxVision"

## Aperçu
Ce projet se concentre sur le développement d'un système de détection d'objets de circulation à l'aide de YOLOv8. L'objectif est d'identifier et de classer avec précision divers participants à la circulation tels que les voitures, les vélos, les motos, les bus et les personnes dans les images.

## Jeu de données
Le jeu de données utilisé pour l'entraînement et la validation est `yusufberksardoan/traffic-detection-project` de Kaggle. Il contient des images annotées avec des boîtes englobantes pour différentes classes d'objets de circulation.

**Source des données :** [Lien vers le jeu de données Kaggle](https://www.kaggle.com/datasets/yusufberksardoan/traffic-detection-project)

## Entraînement du modèle

### Architecture du modèle
Le modèle employé est YOLOv8n, une variante légère de la famille de détection d'objets YOLO (You Only Look Once), connue pour son équilibre entre vitesse et précision.

### Configuration de l'entraînement
*   **Modèle :** `yolov8n.pt` (poids pré-entraînés)
*   **Époques :** 20
*   **Taille de l'image :** 640x640 pixels
*   **Configuration des données :** `data.yaml` (spécifie les chemins d'accès aux images d'entraînement/validation/test et les noms des classes)

## Métriques d'évaluation
Après l'entraînement, le modèle a été évalué sur un ensemble de validation distinct. Les métriques clés sont les suivantes :

*   **mAP (Mean Average Precision) à IoU 0.5-0.95 :** (Valeur de `metrics.box.map`, par exemple, `0.676`)
*   **mAP (Mean Average Precision) à IoU 0.50 :** (Valeur de `metrics.box.map50`, par exemple, `0.913`)
*   **mAP (Mean Average Precision) à IoU 0.75 :** (Valeur de `metrics.box.map75`, par exemple, `0.762`)

### mAP par classe
*   **bicycle :** (Valeur pour la classe 0, par exemple, `0.731`)
*   **bus :** (Valeur pour la classe 1, par exemple, `0.845`)
*   **car :** (Valeur pour la classe 2, par exemple, `0.758`)
*   **motorbike :** (Valeur pour la classe 3, par exemple, `0.553`)
*   **person :** (Valeur pour la classe 4, par exemple, `0.494`)

*(Remarque : Les valeurs ci-dessus sont des exemples et doivent être remplacées par les valeurs réelles obtenues à partir du résultat de l'évaluation du notebook.)*

## Visualisations
Le processus d'entraînement a généré plusieurs tracés pour visualiser les performances, notamment :
*   Matrice de confusion
*   Courbe Précision-Rappel
*   Courbe F1-Confiance
*   Courbe Précision-Confiance

Ces tracés sont enregistrés dans le répertoire `/content/runs/detect/val` et ont été affichés dans le notebook.

## Comment utiliser le modèle entraîné

### Chargement du modèle
Les meilleurs poids du modèle entraîné sont enregistrés sous `best.pt` dans `/content/runs/detect/train/weights/`.

```python
from ultralytics import YOLO
model = YOLO('/content/runs/detect/train/weights/best.pt')
```

### Réalisation de prédictions
Pour faire des prédictions sur de nouvelles images, vous pouvez utiliser la méthode `predict` :

```python
from ultralytics import YOLO
from PIL import Image
import IPython.display

model = YOLO('/content/runs/detect/train/weights/best.pt')
image_path = '/chemin/vers/votre/image.jpg' # Remplacez par le chemin de votre image

results = model.predict(source=image_path, conf=0.25) # conf est le seuil de confiance

for r in results:
    im_array = r.plot() # trace un tableau numpy BGR des prédictions
    im = Image.fromarray(im_array[..., ::-1]) # convertit BGR en image PIL RGB
    IPython.display.display(im)
```

## Configuration et installation

1.  **Cloner ce dépôt (le cas échéant) :**

    ```bash
    git clone <url-du-depot>
    cd <nom-du-depot>
    ```

2.  **Installer les dépendances :**

    ```bash
    pip install ultralytics
    pip install kaggle
    ```

3.  **Clé API Kaggle (pour le téléchargement du jeu de données) :**
    *   Téléchargez votre fichier `kaggle.json` contenant votre clé API Kaggle.
    *   Déplacez-le vers `~/.kaggle/` et définissez les autorisations appropriées :

        ```python
        import os
        from google.colab import files

        # files.upload() # Décommenter si exécuté dans Colab pour télécharger kaggle.json
        os.makedirs("/root/.kaggle", exist_ok=True)
        os.rename("kaggle.json", "/root/.kaggle/kaggle.json")
        os.chmod("/root/.kaggle/kaggle.json", 600)
        ```

4.  **Télécharger le jeu de données :**

    ```bash
    !kaggle datasets download -d yusufberksardoan/traffic-detection-project
    !unzip traffic-detection-project.zip
    ```

## Travaux futurs
*   Expérimenter avec différents modèles YOLOv8 (par exemple, `yolov8m`, `yolov8l`).
*   Intégrer le modèle dans un pipeline de traitement vidéo en temps réel.
*   Explorer des techniques pour améliorer la précision de la détection dans des conditions difficiles (par exemple, faible luminosité, trafic dense).
