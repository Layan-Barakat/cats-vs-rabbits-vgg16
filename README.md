# Cats vs Rabbits — VGG16 (TensorFlow/Keras)

Transfer learning image classifier using VGG16 to distinguish between cats and rabbits.

## What this repo contains
- `cats_vs_rabbits_vgg16.py` – full training + fine-tuning script  
- `requirements.txt` – Python packages needed  
- `predict_single.py` – quick test on one image (optional)  
- `assets/` – plots or demo GIFs (optional)  

The dataset and trained weights are not committed to the repo.

## Folder structure expected locally
```
cat-vs-rabbit/
├─ train-cat-rabbit/
│  ├─ cats/...
│  └─ rabbits/...
├─ val-cat-rabbit/
│  ├─ cats/...
│  └─ rabbits/...
├─ cats_vs_rabbits_vgg16.py
└─ Cats-VS-Rabbits_Output/
   ├─ final.keras
   └─ labels.json
```

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train
```
python cats_vs_rabbits_vgg16.py
```

## Inference (single image)
```
python predict_single.py --img test-images\my_photo.jpg
```

## Model weights
The trained model file is large, so it is not in this repo.

Download `final.keras` from this link:  
[ADD-YOUR-LINK-HERE]

Place it at:
```
Cats-VS-Rabbits_Output\final.keras
```

Ensure `Cats-VS-Rabbits_Output\labels.json` exists (it is created during training).

## Notes
- VGG16 backbone is frozen first, then the last layers are fine-tuned.  
- Light augmentation + caching/prefetch are used.  
- Early stopping + best-model checkpointing are enabled.
