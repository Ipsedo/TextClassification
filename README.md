# TextClassification
_Samuel Berrien_

Un repo pour tester des baselines pour de la classification de documents.

## Requirements
* PyTorch (version à préciser)
* CUDA
* à completer

## DBPedia
Télécharger les données depuis le site officiel de [DBPedia](https://wiki.dbpedia.org/) :
* `instance_types_en.ttl.bz2` sur ce [lien](http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2)
* `long_abstracts_en.ttl.bz2` sur ce [lien](http://downloads.dbpedia.org/2016-10/core-i18n/en/long_abstracts_en.ttl.bz2)
Extraire ces deux fichiers et placer les deux `.ttl` dans le dossier `TextClassification/datasets/`

__Attention__ : Une grosse quantité de RAM est requise (testé avec 32GB de RAM et 12GB de SWAP)

__Résultat__ : Précision de 8% (sans modification), 40\% avec filtrage entre 1000 et 300 avec repondération des classes

## Reuters
Il suffit d'installer le package python NLTK

__Résultats__ :

```
Test : correct = 1782 / 2444, 0.729133

AUC earn = 0.967511
AUC acq = 0.948722
AUC money-fx = 0.795420
AUC grain = 0.805381
AUC crude = 0.938438
AUC trade = 0.872333
AUC interest = 0.931114
AUC wheat = nan
AUC ship = 0.879055
AUC corn = 0.907459
```

## Wiki Dump
Script pour la création du dataset avec LDA à venir

## Fichiers
```
TextClassification/
|-- datasets/ : Données DBPedia + english stopwords
|-- outputs/ : Images résultats des différents benchmark
|-- results/ : Différents résultats sauvegardés
|-- saved/ : Modèle, vocab et class to idx sauvés
|-- analyse.py : Différentes fonctions pour l'analyse de DBPedia
|-- build_dbpedia.py : Prépération du jeu de données DBPedia
|-- main.py : Fonction d'entrainement de DBPedia
|-- models.py : Contient les différents modèles PyTorch expérimentés
|-- old_main.py : Fonction de becnhmark Reuters et Wikipedia étiqueté avec LDA
|-- preprocessing.py : Fonction de pretraitement du texte. Fonction de data augmentation du corpus
|-- user_test_dbpedia.py : Script pour tester en direct le modèle
|-- utils.py : Fonctiones utiles pour l'affichage de l'occupation mémoire.
```