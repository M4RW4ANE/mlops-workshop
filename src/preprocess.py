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
    df = pd.read_csv(file_path, sep=';')  # Utilisation du point-virgule comme séparateur
    
    # Copie pour éviter de modifier les données originales
    data = df.copy()
    
    print(f"Dataset chargé - Shape: {data.shape}")
    print(f"Colonnes: {list(data.columns)}")
    
    # ===== NETTOYAGE DES DONNÉES PROBLÉMATIQUES =====
    
    # Fonction pour nettoyer les valeurs numériques
    def clean_numeric_column(series):
        """Nettoie une colonne numérique en gérant les valeurs multiples et les chaînes"""
        def clean_value(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str):
                # Supprimer les espaces en début/fin
                x = x.strip()
                if x == '' or x.lower() in ['nan', 'null', 'na']:
                    return np.nan
                # Si la valeur contient des espaces (comme '15 16'), prendre le premier nombre
                if ' ' in x:
                    x = x.split()[0]
                # Essayer de convertir en float
                try:
                    return float(x)
                except:
                    return np.nan
            return np.nan
        
        return series.apply(clean_value)
    
    # Nettoyer les colonnes numériques potentiellement problématiques
    numeric_cols_to_clean = ['Age', 'Fare', 'Pclass', 'Sibsp', 'SibSp', 'Parch']
    for col in numeric_cols_to_clean:
        if col in data.columns:
            print(f"Nettoyage de la colonne {col}...")
            # Vérifier s'il y a des valeurs problématiques
            problematic_values = data[col].apply(lambda x: isinstance(x, str) and ' ' in str(x)).sum()
            if problematic_values > 0:
                print(f"  - {problematic_values} valeurs problématiques détectées")
            data[col] = clean_numeric_column(data[col])
    
    # Afficher les statistiques après nettoyage
    print(f"Valeurs manquantes après nettoyage:")
    print(data.isnull().sum())
    
    # ===== NETTOYAGE DES DONNÉES =====
    
    # Suppression des colonnes non utiles pour la prédiction
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Boat', 'Body', 'Home.Dest']
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
    
    # Taille de la famille (gérer les deux variantes de noms de colonnes)
    sibsp_col = 'Sibsp' if 'Sibsp' in data.columns else 'SibSp' if 'SibSp' in data.columns else None
    parch_col = 'Parch' if 'Parch' in data.columns else None
    
    if sibsp_col and parch_col:
        data['Family_Size'] = data[sibsp_col] + data[parch_col] + 1
        
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
        # Vérifier qu'il n'y a pas de valeurs négatives ou aberrantes
        data['Fare'] = data['Fare'].clip(lower=0)
        # Créer les groupes seulement si on a des valeurs valides
        if data['Fare'].notna().sum() > 0:
            try:
                data['Fare_Group'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'], duplicates='drop')
            except:
                # Si qcut échoue, utiliser cut avec des bins manuels
                data['Fare_Group'] = pd.cut(data['Fare'], 
                                          bins=[0, 7.91, 14.45, 31, float('inf')], 
                                          labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # Nettoyage final des données
    print("Nettoyage final des données...")
    
    # Convertir toutes les colonnes numériques en float
    numeric_columns = ['Age', 'Fare', 'Pclass', 'Sibsp', 'SibSp', 'Parch', 'Family_Size']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
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
    
    # Vérifier les colonnes disponibles et définir la variable cible
    print(f"Colonnes disponibles dans le dataset: {list(data.columns)}")
    
    # Rechercher la colonne cible (plusieurs noms possibles)
    target_columns = ['Survived', 'survived', 'target', 'label', 'class']
    target_col = None
    
    for col in target_columns:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        print("Aucune colonne cible trouvée. Colonnes possibles:", target_columns)
        print("Colonnes disponibles:", list(data.columns))
        raise ValueError(f"Aucune colonne cible trouvée parmi {target_columns}. "
                        f"Colonnes disponibles: {list(data.columns)}")
    
    print(f"Utilisation de '{target_col}' comme variable cible")
    y = data[target_col]
    X = data.drop(target_col, axis=1)
    
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
    
    # Vérifier qu'il n'y a pas de valeurs manquantes restantes et nettoyer
    print("Vérification finale des données...")
    
    # Afficher les types de données
    print("Types de données:")
    print(X_train.dtypes)
    
    # Convertir toutes les colonnes object restantes en numérique si possible
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"Conversion de la colonne {col} (type: {X_train[col].dtype})")
            # Vérifier quelques valeurs
            print(f"  Quelques valeurs: {X_train[col].head().tolist()}")
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("Des valeurs manquantes subsistent après le preprocessing")
        print("Valeurs manquantes dans X_train:")
        print(X_train.isnull().sum())
        print("Valeurs manquantes dans X_test:")
        print(X_test.isnull().sum())
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
    df = pd.read_csv(file_path, sep=';')  # Utilisation du point-virgule
    
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