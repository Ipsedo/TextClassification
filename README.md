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
Extraire ces deux fichiers et placer les deux `.ttl` dans `TextClassification/datasets/`

__Attention__ : Une grosse quantité de RAM est requise (testé avec 32GB de RAM et 12GB de SWAP)

## Reuters
Il suffit d'installer le package python NLTK

## Wiki Dump
Script pour la création du dataset avec LDA à venir