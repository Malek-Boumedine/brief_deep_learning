# BRIEF DEEP LEARNING

## 🧠 Contexte Professionnel

Vous êtes Data Scientists dans une ESN (Entreprise de Services Numériques) qui accompagne les opérateurs télécoms dans la réduction de la perte d’abonnés.
Votre nouveau client, TelcoNova, souhaite anticiper les départs de ses clients (churn) afin d’orienter ses campagnes de rétention.
Il met à votre disposition un extrait anonymisé de sa base CRM (le jeu Telco Customer Churn, déjà pré-nettoyé en grande partie) et vous laisse 3 jours pour livrer un premier prototype de modèle de prédiction exploitable en production.
TelcoNova exige un livrable reproductible et facilement intégrable par ses équipes MLOps ; vous travaillerez en binôme, en suivant les bonnes pratiques Git / GitHub.

Les données fournies (**WA_Fn-UseC_-Telco-Customer-Churn.csv**) sont un extrait anonymisé de la base CRM du client, déjà pré-nettoyées en grande partie.

## 🎯 Objectifs & Livrables

### Mission principale :
- Construire un **modèle de prédiction permettant d'anticiper le départ de clients**, basé sur du **Deep Learning**, avec **TensorFlow/Keras ou PyTorch**.


## 🧱 Fonctionnalités Implémentées (MVP)

### 🔍 Préparation des données
- Nettoyage & typage des colonnes
- Encodage systématique des catégorielles 
- Split train / val / test (stratifié)

### 🧠 Modélisation
- MLP implémenté **from scratch** sous TensorFlow/Keras ou PyTorch
- Backprop + optimiseur (Adam ou SGD) 
- Gestion du déséquilibre (pondération ou autre)

### 📦 Suivi & répétabilité
- TensorBoard ou PyTorch Lightning logger 
- Seeds fixés + README détaillant la reproduction 
- ModelCheckpoint pour restaurer le meilleur modèle

### 📈 Évaluation
- Export du **modèle entraîné** (`.h5` ou `.pt`)
- Sauvegarde du **scaler** et des **encoders**
- Script ou notebook d’**inférence reproductible** :
  - Chargement des artefacts
  - Prédiction du churn sur de nouvelles données

### 👷 Collaboration Git
- 1 branch = 1 feature 
- Pull request systématique avec description 


## 🚀 Lancement du projet

### 1. Clôner le dépôt
- bash
  - `git clone https://github.com/Malek-Boumedine/brief_deep_learning.git`
  - `cd brief_deep_learning`

### 2. Création d'un environnement virtuel
- bash :
  - `python -m venv .venv`
- Si vous êtes sur Windows :
  - `source .venv/Scripts/activate`

### 3. Installation des dépendances
- bash
  - `pip install -r requirements.txt`
  - `pip install -r requirements2.txt`

## Informations sur les fichiers
- requirements.txt                     => fichier comprenant les dépendances du projet
- requirements2.txt                    => deuxième fichier comprenant les dépendances du projet
- PreProcessing.py                     => Nettoyage des données, pré-traitement des données
- modelisation.py                      => Implémentation du modèle de prédiction sous TensorFlow
- modelisation_pytorch.py              => notebook de nettoyage, tré-traitement, modélisation et évaluation du modèle sous PyTorch
- utils.py                             => (à compléter)
- WA_Fn-UseC_-Telco-Customer-Churn.csv => dataset
