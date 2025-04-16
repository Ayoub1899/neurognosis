# 🧠 Classification de Tumeurs Cérébrales à partir d'IRM avec Deep Learning

Ce projet vise à détecter automatiquement le type de tumeur cérébrale à partir d'une image d'IRM en utilisant un modèle de deep learning basé sur **ResNet18** et le dataset suivant : **https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri**. Une interface web simple, développée avec **Flask**, permet à l'utilisateur d'importer une image et de recevoir une prédiction en temps réel.

## 📁 Structure du projet


## 🧠 Classes détectées

Le modèle est entraîné pour classer les IRM en 4 catégories :
- `glioma` (gliome)
- `meningioma` (méningiome)
- `notumor` (pas de tumeur)
- `pituitary` (tumeur pituitaire)

## 🚀 Lancer le projet localement

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-utilisateur/ton-projet.git
cd ton-projet
```

### 2. Installer les dépendances
#### Assure-toi d’avoir Python 3.7+ installé, puis :
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application Flask
```bash
python app.py
```
L'application sera disponible sur http://127.0.0.1:5000/




