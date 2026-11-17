# 🔬 Deep Learning Architectures for Binary Skin Cancer Classification

An Applied Deep Learning project that builds a **reproducible, leakage-safe pipeline** to triage dermoscopic skin lesions as **benign or malignant**. It compares a from-scratch CNN baseline with transfer-learning backbones.

## 📌 Highlights

- **Best model: ResNet50 (transfer learning).** Macro-F1 **0.875**, ROC-AUC **0.956**, malignant recall **0.896**
- Transfer learning improved macro-F1 by **+9.4 points** and malignant precision by **+16.7 points** over the from-scratch baseline
- The decision threshold is tuned for **high malignant recall**, because a missed cancer is the costly error
- **Grad-CAM** explainability checks that the model looks at the lesion itself

## 📂 Dataset

**HAM10000** (10,015 dermoscopy images) plus a supplementary ISIC set for class balance. The seven diagnoses were collapsed into a binary benign / malignant target.

- **Lesion-wise, stratified 70/15/15 split** (9,231 / 1,847 / 1,846 images). No lesion appears in more than one split, so there is no leakage.
- A perceptual-hash duplicate audit removed exact duplicates and kept genuine repeat photos of the same lesion.

## 🧠 Methods

| Stage | Details |
|---|---|
| Preprocessing | Resize to 224×224, moderate augmentation that keeps diagnostic cues, normalization computed from the train set |
| Imbalance | Square-root inverse-frequency `WeightedRandomSampler` plus focal loss |
| Baseline | Compact ResNet + Squeeze-and-Excitation CNN trained from scratch |
| Transfer learning | EfficientNet-B0, ResNet50, DenseNet121 |
| Training | Mixed precision, gradient clipping, EMA weights, MixUp |
| Evaluation | Four-view test-time augmentation, threshold chosen on validation only |
| Tuning | Controlled studies of optimizer (Adam / AdamW / SGD), learning rate and weight decay |

## 📊 Results (Test Set)

| Rank | Model | Accuracy | Macro-F1 | ROC-AUC | Malignant Recall | Params | Inference |
|---|---|---|---|---|---|---|---|
| 🥇 | **ResNet50** | **88.8%** | **0.875** | 0.956 | 0.896 | 23.5M | 5.19 ms |
| 🥈 | EfficientNet-B0 | 88.2% | 0.869 | **0.958** | 0.905 | **4.0M** | 5.16 ms |
| 🥉 | DenseNet121 | 87.0% | 0.858 | 0.952 | **0.922** | 7.0M | **5.08 ms** |
| 4 | CNN Baseline (from scratch) | 79.2% | 0.781 | 0.902 | 0.906 | 9.7M | 8.78 ms |

The final configuration was **ResNet50 with AdamW, lr = 1e-4 and weight decay = 0**. EfficientNet-B0 comes very close with **6× fewer parameters**, which makes it the best option for deployment.

## 📁 Project Structure

```
notebooks/
  01_Data_Preparation_and_Baseline.ipynb          # data pipeline, EDA, baseline CNN
  02_Transfer_Learning_and_Model_Comparison.ipynb # architectures + hyperparameter studies
  03_Final_Evaluation_and_Error_Analysis.ipynb    # confusion matrices, ROC/PR, Grad-CAM
src/          # shared config, data, models and training engine
train.py      # reproduce any training run
evaluate.py   # evaluate saved checkpoints
```

## 🚀 Usage

```bash
pip install -r requirements.txt

# Train the final configuration
python train.py --model resnet50 --lr 1e-4 --weight-decay 0.0 --exp-name ResNet50_final

# Evaluate a saved model
python evaluate.py --model resnet50
```

## 🛠️ Tech Stack

Python · PyTorch · torchvision · scikit-learn · Pandas · Matplotlib · Grad-CAM

---

👩‍💻 **Aisha Alsadani**: [LinkedIn](https://linkedin.com/in/aishaalsadani) · [GitHub](https://github.com/aishaalsadani)
