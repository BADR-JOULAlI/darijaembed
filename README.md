# DarijaEmbed

Expériences reproductibles de sentence embeddings pour la darija marocaine :
similarité sémantique, variation orthographique, Arabizi et code-switching.

**État : première implémentation. Aucun gain scientifique n'est encore mesuré.**
Le modèle de départ est un bi-encodeur partagé :
[paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
qui produit des vecteurs de 384 dimensions. La dimension dépend du modèle choisi,
elle n'est pas imposée dans le code.

## Installation

Python 3.11 ou supérieur. Depuis la racine du dépôt :

```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate
# PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -e ".[ml,dev]"
darijaembed doctor
```

Pour une GPU NVIDIA, installer d'abord la version de PyTorch adaptée au pilote
avec le sélecteur [officiel](https://pytorch.org/get-started/locally/).
`doctor` affiche la version de PyTorch, CUDA et les GPU détectées.
`device = "auto"` sélectionne CUDA si disponible, sinon CPU.
Choisir `device = "cuda"` pour exiger la GPU et échouer si elle est inaccessible.
La compatibilité et la consommation mémoire de la RTX 5070 restent à vérifier
sur la machine d'entraînement ; les valeurs de configuration sont un point de départ.

Pour tester seulement le chargement local et la préparation des données,
`python -m pip install -e .` suffit : aucune dépendance ML n'est importée par cette commande.

## Démarrer

Vérifier d'abord le pipeline avec des exemples synthétiques inclus dans le dépôt :

```bash
darijaembed prepare --config configs/local.toml
```

Ce fichier contient des phrases écrites pour les tests. Elles ne constituent
ni un corpus de recherche ni un benchmark validé. `train` avec ce fichier
télécharge encore le modèle configuré ; seul `prepare` est entièrement hors ligne.

Pour le smoke test arabe–Arabizi sur 10 000 lignes et 100 étapes :

```bash
darijaembed prepare --config configs/smoke.toml
# Lire runs/smoke/data/audit.json avant l'entraînement.
darijaembed train --config configs/smoke.toml --prepared-dir runs/smoke/data
```

`train` peut aussi préparer les données directement si `--prepared-dir` est omis.
L'entraînement optimise `MultipleNegativesRankingLoss` avec des négatifs du batch,
et conserve la colonne `negative` lorsqu'elle est configurée.
Les paramètres du Trainer suivent la
[documentation Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html).
L'accumulation des gradients modifie les mises à jour de l'optimiseur ; elle
n'agrandit pas le groupe de négatifs partagé entre les microbatches.
La perte de validation dépend de la composition des batches ; elle ne remplace
pas une mesure STS. Un batch réduit à une paire apporte peu ou pas de signal
contrastif selon la présence de négatifs explicites.

Pour les expériences sémantiques, adapter `configs/semantic.toml`, accepter
les conditions des datasets sur Hugging Face et se connecter avec `hf auth login`.
Comparer la baseline et le modèle entraîné avec exactement la même configuration :

```bash
darijaembed evaluate --config configs/semantic.toml
darijaembed prepare --config configs/semantic.toml
darijaembed train --config configs/semantic.toml --prepared-dir runs/semantic/data
darijaembed evaluate --config configs/semantic.toml --model runs/semantic/model
```

Réserver le benchmark AtlasIA à la comparaison finale ; choisir les paramètres
avec la validation issue de l'entraînement. Ne pas consulter les scores du test
à chaque essai. Les deux rapports d'évaluation sont conservés séparément.
Un résultat sur des fixtures ou une seule paire ne démontre pas une amélioration.

## Configuration et architecture

```text
configs/                 # Expériences TOML : smoke, semantic, local
src/darijaembed/
  config.py              # Paramètres typés et validation
  data.py                # Sources JSONL/CSV/HF, audit et séparation
  training.py            # Fine-tuning contrastif et provenance
  evaluation.py          # Similarité cosinus, Spearman et Pearson
  cli.py                 # prepare, train, evaluate, doctor
tests/                   # Tests locaux et fixtures synthétiques
```

Changer un modèle, une source, les colonnes ou les paramètres d'entraînement
se fait dans TOML. Les chemins locaux et `output_dir` sont relatifs au fichier
de configuration. `--prepared-dir` et `--model` sont relatifs au terminal.
Les modèles qui nécessitent des préfixes peuvent utiliser `model.query_prompt`
et `model.document_prompt`, appliqués aussi bien au training qu'à l'évaluation.
Pour une autre baseline, copier une configuration, changer `model.name`,
ses préfixes et `output_dir`, puis lancer `evaluate` sur le même test.

La préparation normalise uniquement Unicode NFC et les espaces. Elle retire
les textes vides, les paires identiques et les doublons de paires positives,
y compris les orientations inversées ; la première occurrence est conservée.
Tous les exemples reliés par une phrase identique restent dans le même groupe,
y compris via les négatifs. La séparation évite les fuites exactes entre train
et validation ; elle ne détecte pas les paraphrases ou les documents sources communs.
Un dataset trop connecté échoue avec un message explicite au lieu de produire
une séparation incorrecte. La fraction obtenue figure dans l'audit.

Chaque entraînement produit `run.json` et `model/`. La préparation conserve
les paires, la configuration résolue et les empreintes SHA256 sous `data/`.
Les métriques STS et la provenance sont enregistrées sous `evaluation/`.
Un dossier contenant un entraînement précédent est refusé : utiliser un nouvel
`output_dir` par expérience. Les seeds, versions et empreintes facilitent la
reproduction ; elles ne garantissent pas des résultats GPU identiques bit à bit.
Pour figer les ressources distantes, renseigner `model.revision` et
`dataset.revision` avec leurs SHA Hugging Face avant une expérience publiée.
Les dépendances utilisent des plages de versions ; conserver aussi
`python -m pip freeze` dans le dossier de l'expérience.

## Données et périmètre scientifique

| Source | Usage prévu | Accès et provenance |
| --- | --- | --- |
| [OiQ/daa-pairs](https://huggingface.co/datasets/OiQ/daa-pairs) | Alignement arabe–Arabizi | Colonnes `arabic`, `arabizi`, `mixte` ; carte Apache-2.0 |
| [Sentence-Transformers-Morocco-Darija](https://huggingface.co/datasets/BounharAbdelaziz/Sentence-Transformers-Morocco-Darija) | Paires/triplets sémantiques | Accès conditionnel ; traduction automatique de MLDR/MIRACL, qualité à auditer |
| [AtlasIA STS](https://huggingface.co/datasets/atlasia/Morocco-Darija-Sentence-Embedding-Benchmark) | Test final STS | Accès conditionnel ; carte CC-BY-NC-3.0 ; données générées puis annotées humainement |

Vérifier le schéma et les conditions de chaque source avant l'expérience.
Le dépôt ne redistribue aucun de ces datasets ni leurs poids de modèles.
La licence MIT concerne uniquement le code original du dépôt.

Cette version implémente la préparation, le fine-tuning et les corrélations STS.
Les métriques de recherche documentaire (Recall@k, MRR, nDCG), les sous-ensembles
de robustesse annotés, l'audit humain des traductions, les comparaisons multi-seeds
et les intervalles de confiance restent à développer. Il n'y a pas encore de
modèle DarijaEmbed publié ni de revendication de performance.

## Vérifier le code

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

La CI exécute les tests sur Python 3.11 et 3.12 sans télécharger de modèle.
Les tests unitaires ML utilisent des doubles. Un job CPU séparé vérifie aussi
la préparation, l'évaluation, deux étapes d'entraînement et le rechargement
avec un petit Transformer aléatoire créé localement. Il ne mesure pas la qualité
linguistique. Pour exécuter ce test avec les dépendances ML installées :

```bash
# PowerShell : $env:DARIJAEMBED_RUN_ML_TESTS = "1"
DARIJAEMBED_RUN_ML_TESTS=1 pytest -q tests/test_pipeline_integration.py
```

La disponibilité réelle de CUDA et la qualité du modèle se vérifient avec une
expérience séparée.
