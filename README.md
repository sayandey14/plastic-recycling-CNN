# Plastic Sorting: drinking-waste classifier

Classifies photos into AluCan, Glass, HDPEM and PET (Kaggle `arkadiyhacks/drinking-waste-classification`).

| | Original notebook | This version |
|---|---|---|
| Validation accuracy (same 962 images) | 47.82% | **99.17%** (99.06% without TTA) |

## Files
- `plastic_sorting.ipynb`: Colab notebook. Open it in Colab, select a GPU runtime and run all cells. It downloads the data with `kagglehub`, so no `kaggle.json` is needed.
- `train.py`: training script; the notebook writes this same file into the Colab runtime.
- `predict.py`: classify your own photos: `python predict.py artifacts/model.pt photo.jpg`
- `artifacts/`: the trained model (`model.pt`, ConvNeXt-Tiny at 256 px), `labels.txt` and `results.json` (per-epoch history and confusion matrix).

## Run locally
```
pip install torch torchvision timm scikit-learn pillow kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('arkadiyhacks/drinking-waste-classification'))"
python train.py --data <that path>/Images_of_Waste/rawimgs --workers 2
```
On an 8 GB Mac this takes about 80 minutes (about 6.5 min/epoch on MPS). If memory is tight, add `--grad-checkpointing`.
