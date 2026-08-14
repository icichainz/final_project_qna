# Système de Questions-Réponses

## Développer un système capable de répondre à des questions formulées en langage naturel, basé sur un ensemble de documents donnés

---

![Logo du Projet](https://via.placeholder.com/150)

---

**Auteur :** KOUDAYA KOKOU ABEL

**Date :** 30 Aout 2024

**Version :** 1.0

---

# Table des matières

[TOC]

---

# I. Introduction
- **Contexte du projet** : Vous travaillez sur un projet de développement d'un système de questions-réponses (Q&A) intégré dans une application, probablement pour permettre aux utilisateurs de poser des questions et d'obtenir des réponses pertinentes et contextuelles. Ce projet semble faire partie d'une initiative plus large visant à offrir des fonctionnalités d'assistance ou d'apprentissage automatisé, où les utilisateurs peuvent interagir avec le système en langage naturel pour obtenir des informations ou des solutions à leurs problèmes.
- **Objectifs du système de questions-réponses:**
Le système de questions-réponses a pour objectifs principaux : Faciliter l'accès à l'information : 
    1. Permettre aux utilisateurs d'obtenir rapidement des réponses à leurs questions sans avoir à parcourir des bases de données ou des documents volumineux.

    2. Fournir des réponses contextuelles et pertinentes : Utiliser des techniques d'apprentissage automatique et de traitement du langage naturel pour fournir des réponses qui tiennent compte du contexte de la question posée.

    3. Améliorer l'efficacité : Réduire le temps et les efforts nécessaires pour trouver des informations précises, augmentant ainsi l'efficacité du processus de prise de décision ou d'apprentissage.

    4. Apprentissage continu : Permettre au système d'améliorer ses réponses au fil du temps grâce à des mécanismes de feedback et d'apprentissage supervisé ou non supervisé.
- **Aperçu des technologies utilisées (notamment Langchain):** Les technologies utilisé sont : 
    1. Lanchain : Qui est un cadre développment pour créer des applications basées sur des modèles de language, en particulier celles qui impliques des chaines de traitement complexes.

# II. Méthodologie
## 1. Collecte de données
### Composants clés :
- **Chainlit pour l'interface utilisateur** : Offre une interface conviviale pour le téléchargement de fichiers, facilitant l'interaction entre l'utilisateur et le système.
- **Gestion de plusieurs formats de fichiers** : Prise en charge des formats TXT, PDF et DOCX, élargissant la gamme de documents pouvant être traités.
- **Limite de taille de fichier** : Restriction à 20 MB pour gérer la charge du système et les temps de traitement.

### Implémentation :
```python
@cl.on_chat_start
async def on_chat_start():
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain",
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_size_mb=20,
            timeout=180,
        ).send()
```

### Importance :
La collecte efficace des données est cruciale car elle détermine la base de connaissances sur laquelle le système s'appuiera. Une interface utilisateur intuitive et la prise en charge de divers formats de fichiers augmentent la flexibilité et l'utilité du système.

## 2. Prétraitement des données
### Composants clés :
- **Fonctions spécifiques de traitement** : Les fonctions `process_pdf` et `process_word` extraient le texte des différents formats de fichiers.
- **RecursiveCharacterTextSplitter** : Cette méthode de découpage du texte permet de créer des segments de taille cohérente tout en préservant le contexte.
- **Paramètres de découpage** : Les valeurs de `chunk_size` et `chunk_overlap` influencent directement la granularité de l'information et la préservation du contexte.

### Implémentation :
```python
def process_pdf(path_to_file):
    text= " "
    reader = PdfReader(path_to_file)
    for page in reader.pages:
        text += page.extract_text() +"\n\n"
    return text  

def process_word(path_to_file):
    text =  docxpy.process(path_to_file)
    return text  

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(text)
```

### Importance :
Un prétraitement efficace assure que les données sont dans un format optimal pour l'indexation et la recherche. Le découpage approprié du texte est crucial pour maintenir la cohérence sémantique tout en permettant une recherche précise.

## 3. Indexation des documents
### Composants clés :
- **HuggingFaceEmbeddings** : Cette bibliothèque fournit des modèles d'embeddings de pointe pour transformer le texte en vecteurs numériques.
- **Chroma vectorstore** : Chroma offre une solution efficace pour stocker et rechercher des embeddings vectoriels.
- **Métadonnées** : L'ajout de métadonnées à chaque chunk facilite le suivi et la récupération des sources.

### Implémentation :
```python
metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
embeddings = HuggingFaceEmbeddings()
docsearch = await cl.make_async(Chroma.from_texts)(
    texts, embeddings, metadatas=metadatas
)
```

### Importance :
Une indexation efficace est essentielle pour des recherches rapides et précises. Les embeddings capturent la sémantique du texte, permettant des comparaisons de similarité plus sophistiquées que la simple correspondance de mots-clés.

## 4. Développement du modèle de questions-réponses
### Composants clés :
- **ConversationalRetrievalChain** : Cette chaîne de Langchain combine la recherche de documents pertinents avec la génération de réponses cohérentes.
- **ChatAnthropic (claude-3-5-sonnet-20240620)** : Ce modèle de langage avancé permet de générer des réponses naturelles et contextuellement appropriées.
- **Système de mémoire** : La mémoire de conversation permet de maintenir le contexte sur plusieurs échanges.

### Implémentation :
```python
chain = ConversationalRetrievalChain.from_llm(
    ChatAnthropic(model="claude-3-5-sonnet-20240620",),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    memory=memory,
    return_source_documents=True,
)
```

### Importance :
Le cœur du système de questions-réponses réside dans sa capacité à comprendre les questions, à récupérer les informations pertinentes et à formuler des réponses cohérentes. L'utilisation d'un modèle de langage avancé et d'une chaîne de traitement sophistiquée permet d'atteindre cet objectif.

## 5. Intégration de Langchain
### Composants clés :
- **ConversationalRetrievalChain** : Gère le flux de travail global du système de questions-réponses.
- **Chroma** : S'intègre avec Langchain pour fournir des capacités de recherche vectorielle.
- **RecursiveCharacterTextSplitter** : Utilisé pour le prétraitement du texte dans le pipeline Langchain.

### Importance :
Langchain fournit un cadre cohérent pour construire des applications de traitement du langage naturel. Son intégration facilite le développement de systèmes complexes de questions-réponses en fournissant des composants modulaires et interopérables.

## 6. Test et évaluation
### Composants clés :
- **Gestion des interactions utilisateur avec Chainlit** : Permet de simuler des scénarios d'utilisation réels.
- **Récupération des sources** : Fournit une traçabilité des réponses générées.
- **AsyncLangchainCallbackHandler** : Permet un suivi asynchrone des opérations de la chaîne Langchain.

### Implémentation :
```python
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
```

### Importance :
Une évaluation rigoureuse est cruciale pour assurer la qualité et la fiabilité du système. La capacité à tracer les sources des réponses permet une validation manuelle et potentiellement une amélioration continue du système.

## 7. Déploiement
### Composants clés :
- **Interface web Chainlit** : Fournit une plateforme interactive pour l'utilisation du système.
- **Affichage des réponses et des sources** : Améliore la transparence et la confiance dans les réponses générées.

### Importance :
Un déploiement efficace rend le système accessible et utilisable. L'interface utilisateur joue un rôle crucial dans l'adoption et l'utilisation effective du système, en facilitant l'interaction et en fournissant des informations claires sur les réponses générées.

# III. Résultats
- Performance globale du système
- Exemples de questions-réponses réussies
- Limites identifiées

# IV. Discussion
- Analyse des points forts et des faiblesses du système
- Comparaison avec d'autres approches ou systèmes existants
- Pistes d'amélioration future

# V. Conclusion
- Récapitulatif des réalisations
- Impact potentiel du système
- Perspectives d'évolution du projet

# VI. Références
- Liste des sources, articles et outils cités dans le rapport

# VII. Annexes
- Détails techniques supplémentaires
- Exemples de code (si pertinent)
- Captures d'écran de l'interface utilisateur
