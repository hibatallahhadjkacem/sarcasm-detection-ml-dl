#  Détection du Sarcasme dans les Tweets

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![VS Code](https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

##  Description
Projet de Machine Learning et Deep Learning visant à détecter automatiquement 
si un tweet en anglais est sarcastique ou non.
Développé dans le cadre d'un Mini-Projet académique à la Faculté des Sciences 
de Bizerte, Université de Carthage (CI1 - 2025).

##  Objectif
Construire un système de classification capable de reconnaître le sarcasme 
à partir du contenu des messages (tweets en anglais).

##  Dataset : iSarcasmEval_En
- **Source** : [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval)
- **Train** : `train.En.csv` → entraînement et validation
- **Test** : `task_A_En_test.csv` → test final
- **Variable cible** : `sarcastic` (1 = sarcastique, 0 = non sarcastique)

##  Structure du projet
```
ProjetML/
├── ProjetMLDV.ipynb          # Notebook principal complet (ML + DL)
├── train.En.csv              # Dataset d'entraînement
├── task_A_En_test.csv        # Dataset de test
├── tfidf_vectorizer.pkl      # Vectoriseur TF-IDF sauvegardé
├── mon_svm.pkl               # Modèle SVM sauvegardé
├── model_knn_base.pkl        # Modèle KNN sauvegardé
├── lstm_sarcasm_model.h5     # Modèle LSTM sauvegardé
└── tokenizer.pkl             # Tokenizer sauvegardé
```

##  Étapes du projet

###  Exploration et Analyse des Données (EAD)
- Chargement avec pandas
- Analyse des colonnes (.info(), .head(), .describe())
- Vérification de la distribution des classes
- Visualisations : histogrammes, WordCloud

###  Prétraitement du texte
- Conversion en minuscules
- Suppression de la ponctuation, liens, hashtags, mentions
- Suppression des stopwords
- Création de la colonne `text_clean`

###  Vectorisation
- **TF-IDF Vectorizer** (Machine Learning)
- **Word Embeddings GloVe** (LSTM)
- **Embeddings Transformer** (BERT)

###  Partie Machine Learning
Modèles testés avec et sans GridSearch :
-  Régression Logistique
-  SVM (LinearSVC)
-  Random Forest
-  Naive Bayes (SMOTE)
-  KNN (SMOTE)

###  Partie Deep Learning
-  LSTM (Word Embeddings GloVe)
-  BERT (bert-base-uncased)

##  Résultats

###  Machine Learning
| Modèle | Vectorisation | Accuracy | Précision | Rappel | F1-Score | AUC |
|---|---|---|---|---|---|---|
| Régression Logistique | TF-IDF | 0.5706 | 0.2924 | 0.5087 | 0.3713 | 0.5775 |
| SVM (LinearSVC) | TF-IDF | 0.5994 | 0.3034 | 0.4682 | 0.3682 | 0.5600 |
| KNN (SMOTE) | TF-IDF | 0.5447 | 0.2813 | 0.5318 | 0.3680 | 0.5556 |
| Naive Bayes (SMOTE) | TF-IDF | 0.6023 | 0.3042 | 0.4624 | 0.3670 | 0.5724 |
| Random Forest | TF-IDF | 0.6744 | 0.3197 | 0.2717 | 0.2938 | 0.5800 |

 **Meilleur modèle ML : Régression Logistique** (F1-Score : 0.3713, AUC : 0.5775)

###  Deep Learning
| Modèle | Vectorisation | Accuracy | Précision | Rappel | F1-Score | AUC |
|---|---|---|---|---|---|---|
| BERT | Embeddings Transformer | 0.7205 | 0.4371 | 0.4220 | 0.4294 | 0.66 |
| LSTM | Word Embeddings GloVe | 0.4121 | 0.2736 | 0.8208 | 0.4104 | 0.57 |

 **Meilleur modèle DL : BERT** (Accuracy : 0.7205, AUC : 0.66)

##  Conclusion
- Le déséquilibre du dataset limite les performances des modèles ML classiques
- BERT surpasse tous les modèles avec une AUC de 0.66
- Les modèles DL donnent leur plein potentiel avec des datasets plus volumineux

##  Comment lancer le projet
1. Cloner le repository
```bash
git clone https://github.com/hibatallah-hadjkacem/sarcasm-detection-ml-dl
```
2. Installer les dépendances
```bash
pip install pandas scikit-learn tensorflow transformers torch nltk wordcloud matplotlib seaborn imbalanced-learn
```
3. Ouvrir le notebook principal
```bash
jupyter notebook ProjetMLDV.ipynb
```

##  Réalisé par
- Hibat Allah Hadj Kacem
- Amal Ktiti

##  Contexte académique
- **Établissement** : Faculté des Sciences de Bizerte, Université de Carthage
- **Classe** : CI1
- **Matière** : Machine Learning
- **Année** : 2025
