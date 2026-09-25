# Jaguar Re-Identification Challenge

A computer vision project for identifying individual jaguars from wildlife photographs using image embeddings and similarity-based retrieval.

**Kaggle competition:** [https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge](https://www.kaggle.com/competitions/jaguar-re-id)

## Project overview

The goal is to tell whether two photographs show the same jaguar. Instead of predicting a name directly for each test image, the model learns an embedding for every image and compares those embeddings using cosine similarity.

This repository grew from my first full image re-identification competition. The main focus was keeping the workflow simple, modular, and easy to experiment with.

## Dataset

The training set contains labeled photographs of 31 individual jaguars. The test set contains image pairs that must be assigned a similarity score between 0 and 1.

The images vary in pose, lighting, sharpness, viewpoint, and segmentation quality. Some identities also have many more photographs than others.

## Important data availability note

The competition images are **not included in this repository**.

The data belongs to the competition organizers and is subject to Kaggle competition rules. To reproduce the experiments, download the data directly from Kaggle after accepting the competition terms and conditions.

## Modeling approach

The project uses a configurable image re-identification pipeline:

- background-aware preprocessing using the provided alpha masks
- pretrained image backbones
- learned image embeddings
- linear and ArcFace heads explored during experimentation
- stratified cross-validation
- identity-balanced retrieval validation
- fold ensembles using averaged normalized embeddings

Different image resolutions, embedding sizes, backbones, training lengths, and heads were compared without changing the overall pipeline structure.

## Key results

- Final competition score: **0.88**
- Increasing training time improved performance substantially.
- ConvNeXt outperformed the earlier EfficientNet-based baseline.
- A 512-dimensional embedding performed better than the smaller embedding in later experiments.
- A simple linear classification head performed better than ArcFace in my experiments.
- Fold ensembles provided an additional improvement.

## Repository structure

```text
.
├── configs/
│   └── baseline.yaml
├── notebooks/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

The `notebooks/` folder is intentionally empty here so the original Kaggle notebooks can be added separately.

## Organizers and Citation

The competition was organized by **KINETO.AI / Kineto UG** and the **Jaguar Identification Project**, with data connected to the Pantanal Jaguar ID effort.

@misc{jaguarreidentification-kaggle-2026, author = {Rueda-Toicen, Antonio and Martin, Abigail}, title = {Jaguar Re-identification Kaggle Challenge}, year = {2026}, url = {https://www.kaggle.com/competitions/jaguar-re-id} }

## Disclaimer

This repository is an educational portfolio project based on a Kaggle competition. It is not an official repository of Kaggle, Kineto UG, or the Jaguar Identification Project. Competition data and wildlife images are not redistributed here.
