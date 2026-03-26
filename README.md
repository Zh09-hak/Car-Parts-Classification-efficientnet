# 🚗 Car Parts Classification using EfficientNet

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

---

## 📌 Overview

This project classifies car parts into 40 categories using transfer learning with EfficientNet.

---

## 📊 Dataset

https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes

⚠️ Dataset is not included due to size limitations.

---

## 🧠 Model

* EfficientNetB0 (pretrained on ImageNet)
* Frozen backbone + custom classifier
* Transfer learning approach

---

## 📈 Results

* Training Accuracy: 96.9%
* Validation Accuracy: 94.5%

---

## ⚠️ Important Insight

Fine-tuning the entire model reduced performance.

This suggests:

* Dataset size is limited
* Overfitting occurred during full fine-tuning

---

## ⚙️ Project Structure

```bash
project/
├── src/
├── data/ (not included)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

---

## 🔮 Example Predictions

* Brake Pad → class 4
* Radiator Fan → class 28
* Ignition Coil → class 18

---

## 💡 Future Improvements

* Partial fine-tuning
* Data augmentation
* Larger dataset
* EfficientNetB3/B4

---

## 👨‍💻 Author

https://github.com/Zh09-hak
