# Projet de fin de module — Deep Learning (EMSI 2025–2026)

Conception, implémentation, comparaison et analyse critique de modèles de deep learning pour
données **tabulaires**, **images** et **séquences**, sous PyTorch.

Le projet est entièrement contenu dans **trois notebooks `.ipynb` autonomes** : chaque notebook importe
ses données automatiquement, contient sa propre configuration et tout son code (aucun fichier `.py`
externe n'est requis pour l'exécution).

## Notebooks

| Notebook | Partie | Dataset | Contenu |
|---|---|---|---|
| `P1_MLP_HeartDisease.ipynb` | I — MLP / tabulaire | **Heart Disease (Cleveland, UCI)** | 2 MLP (Sequential + classe), 3 initialisations, inspection des paramètres, save/load, CPU/GPU, métriques + matrice de confusion |
| `P2_CNN_EMNIST.ipynb` | II — CNN / images | **EMNIST Letters (26 classes)** | calculs manuels, corr2d / pooling *from scratch* vs PyTorch, LeNet paramétré, étude padding/stride/pooling/filtres/conv 1×1, feature maps, MLP vs CNN |
| `P3_RNN_Seq2Seq_Tatoeba.ipynb` | III — RNN/LSTM/GRU/Seq2Seq / texte | **Tatoeba deu→eng** | modèle de langage + perplexité, comparaison RNN/LSTM/GRU, BPTT + gradient clipping, Seq2Seq encodeur-décodeur, décodage glouton + beam search, BLEU + question transversale finale |

> Les datasets retenus diffèrent volontairement de ceux cités en exemple dans le cahier de charge
> (Wine/Breast Cancer/Adult ; MNIST/Fashion/CIFAR ; IMDb/fra-eng), tout en restant des jeux **réels**.

## Exécution sur Google Colab (recommandé)

1. Ouvrir chaque notebook sur [Google Colab](https://colab.research.google.com/).
2. Activer le GPU : **Exécution → Modifier → Paramètres du notebook → Accélérateur matériel : GPU**.
3. **Exécuter toutes les cellules** (`Exécution → Tout exécuter`).

Chaque notebook installe lui-même ses dépendances spécifiques via `%pip` (`ucimlrepo`, `sacrebleu`) ;
`torch`, `torchvision`, `scikit-learn`, `pandas` et `matplotlib` sont déjà présents sur Colab. Les données
sont téléchargées automatiquement (UCI / torchvision / Tatoeba), avec **miroirs de secours**.

## Exécution locale

```bash
pip install torch torchvision scikit-learn pandas matplotlib ucimlrepo sacrebleu
jupyter notebook
```

Les chemins de sauvegarde basculent automatiquement de `/content/...` (Colab) vers des dossiers locaux.

## Principes d'ingénierie

- **Non hardcodé** : chaque notebook commence par une cellule `CONFIG` (`@dataclass`) regroupant *toute* la
  configuration (dataset, splits, seed, device, hyperparamètres, chemins). Le reste du code ne fait que
  référencer `CFG` — il suffit de modifier la config pour changer le comportement.
- **Reproductibilité** : `set_seed()` et sélection automatique du *device* (`try_gpu()`).
- **Rigueur** : préparation des données sans fuite (pipelines ajustés sur le train), métriques adaptées,
  et chaque figure/tableau est interprété(e) en markdown.

## Réglage du temps de calcul

Les plafonds par défaut (échantillons, époques, nombre de paires) sont fixés dans la cellule `CONFIG` de
chaque notebook pour un compromis temps/qualité. Sur GPU Colab, on peut les augmenter
(`max_train_samples`, `epochs`, `max_pairs`, …) pour de meilleurs résultats.
