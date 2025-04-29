import os
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

#  Forcer NLTK à utiliser le bon dossier
nltk.data.path.insert(0, "/Users/emnabouzguenda/nltk_data")

#  Vérifier que `punkt` est bien installé
punkt_path = "/Users/emnabouzguenda/nltk_data/tokenizers/punkt"
if not os.path.exists(punkt_path):
    print(" `punkt` introuvable, installation en cours...")
    nltk.download('punkt', download_dir="/Users/emnabouzguenda/nltk_data")
else:
    print("✅ `punkt` trouvé !")

# 📌 Vérifier et télécharger les autres ressources si besoin
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"📥 Téléchargement de {resource}...")
        nltk.download(resource, download_dir="/Users/emnabouzguenda/nltk_data")

print("NLTK Path:", nltk.data.path)

#  Charger le dataset
df = pd.read_csv('responses_export.csv')

#  Fonction de nettoyage du texte
def clean_text(text):
    if pd.isnull(text):  # Gérer les valeurs manquantes
        return ""
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces supplémentaires
    sentences = sent_tokenize(text)  # Tokenisation phrase par phrase
    words = [word_tokenize(sentence) for sentence in sentences]  # Tokenisation mot par mot
    words = [item for sublist in words for item in sublist]  # Aplatir la liste
    words = [word for word in words if word not in stopwords.words('english')]  # Supprimer les stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatisation
    return ' '.join(words)

#  Appliquer le nettoyage aux réponses
df['cleaned_response'] = df['response'].apply(clean_text)

#  Sauvegarder les données nettoyées
df.to_csv('cleaned_responses.csv', index=False)

print("✅ Data cleaned and saved as 'cleaned_responses.csv'")

