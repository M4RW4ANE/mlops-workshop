import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess(file_path, test_size=0.2, random_state=42):
    """
    Préprocesse le dataset Titanic pour l'entraînement d'un modèle de classification.
    
    Args:
        file_path (str): Chemin vers le fichier CSV du dataset Titanic
        test_size (float): Proportion des données pour le test (par défaut 0.2)
        random_state (int): Graine aléatoire pour la reproductibilité
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    
    # Chargement des données
    df = pd.read_csv(file_path)
    
    # Copie pour éviter de modifier les données originales
    data = df.copy()
    
    # ===== NETTOYAGE DES DONNÉES =====
    
    # Suppression des colonnes non utiles pour la prédiction
    columns_to_drop = ['PassengerId', 'Name', 'Ticket']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # ===== TRAITEMENT DES VALEURS MANQUANTES =====
    
    # Age: remplir par la médiane
    if 'Age' in data.columns:
        age_imputer = SimpleImputer(strategy='median')
        data['Age'] = age_imputer.fit_transform(data[['Age']])
    
    # Cabin: créer une variable binaire indiquant si la cabine est renseignée
    if 'Cabin' in data.columns:
        data['Has_Cabin'] = data['Cabin'].notna().astype(int)
        data = data.drop('Cabin', axis=1)
    
    # Embarked: remplir par le mode (valeur la plus fréquente)
    if 'Embarked' in data.columns:
        embarked_mode = data['Embarked'].mode()[0] if not data['Embarked'].mode().empty else 'S'
        data['Embarked'] = data['Embarked'].fillna(embarked_mode)
    
    # ===== CRÉATION DE NOUVELLES VARIABLES =====
    
    # Taille de la famille
    if 'SibSp' in data.columns and 'Parch' in data.columns:
        data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
        
        # Catégorie de famille
        data['Family_Category'] = pd.cut(data['Family_Size'], 
                                       bins=[0, 1, 4, float('inf')], 
                                       labels=['Solo', 'Small', 'Large'])
    
    # Extraction du titre du nom (si la colonne Name existe encore)
    if 'Name' in df.columns:
        titles = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        # Regroupement des titres rares
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        data['Title'] = titles.map(title_mapping).fillna('Rare')
    
    # Binning de l'âge en groupes
    if 'Age' in data.columns:
        data['Age_Group'] = pd.cut(data['Age'], 
                                 bins=[0, 12, 18, 35, 60, float('inf')], 
                                 labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Binning du prix du billet
    if 'Fare' in data.columns:
        # Remplir les valeurs manquantes de Fare par la médiane
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        data['Fare_Group'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # ===== ENCODAGE DES VARIABLES CATÉGORIELLES =====
    
    # Variables catégorielles à encoder
    categorical_columns = ['Sex', 'Embarked', 'Family_Category', 'Age_Group', 'Fare_Group']
    if 'Title' in data.columns:
        categorical_columns.append('Title')
    
    # Encodage one-hot pour les variables catégorielles
    for col in categorical_columns:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop(col, axis=1)
    
    # ===== SÉPARATION DES FEATURES ET DU TARGET =====
    
    # Définir la variable cible
    if 'Survived' not in data.columns:
        raise ValueError("La colonne 'Survived' n'existe pas dans le dataset")
    
    y = data['Survived']
    X = data.drop('Survived', axis=1)
    
    # ===== DIVISION TRAIN/TEST =====
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # ===== NORMALISATION DES DONNÉES =====
    
    # Identifier les colonnes numériques pour la normalisation
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
    
    # Vérifier qu'il n'y a pas de valeurs manquantes restantes
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("Attention: Des valeurs manquantes subsistent après le preprocessing")
        # Remplir les valeurs manquantes restantes par 0
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    print(f"Preprocessing terminé:")
    print(f"- Forme des données d'entraînement: {X_train.shape}")
    print(f"- Forme des données de test: {X_test.shape}")
    print(f"- Nombre de features: {X_train.shape[1]}")
    print(f"- Colonnes finales: {list(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test


def get_feature_info(file_path):
    """
    Fonction utilitaire pour explorer les données avant preprocessing.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
    """
    df = pd.read_csv(file_path)
    
    print("=== INFORMATIONS SUR LE DATASET ===")
    print(f"Forme du dataset: {df.shape}")
    print("\n=== COLONNES ET TYPES ===")
    print(df.dtypes)
    print("\n=== VALEURS MANQUANTES ===")
    print(df.isnull().sum())
    print("\n=== PREMIÈRES LIGNES ===")
    print(df.head())
    print("\n=== STATISTIQUES DESCRIPTIVES ===")
    print(df.describe())
    
    return df