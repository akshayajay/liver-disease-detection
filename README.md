# AI-Driven Liver Disease Diagnosis

> Multi-modal fusion of medical imaging, EHR data, and lab values for automated liver disease classification across 23 conditions.

---

## Overview

This system diagnoses liver conditions by fusing three data modalities:

| Modality | Method |
|---|---|
| Medical images (CT / MRI / Ultrasound) | Vision Transformer (ViT-Base, `google/vit-base-patch16-224-in21k`) |
| Electronic Health Records (EHR) | TF-IDF vectorisation |
| Blood lab values (ALT, AST, ALP, Bilirubin) | Regex-parsed numeric features |

The fused feature vector is classified by an XGBoost model trained on synthetically generated patient data. For each predicted condition the app returns a structured clinical summary and suggested management plan.

**This project was submitted for a patent** — the novel contribution is the application of Vision Transformers (ViT) to synthetically generated liver medical images in a multi-modal diagnostic pipeline, achieving **85–93% precision** across 23 liver conditions.

---

## Conditions Diagnosed (23)

Benign lesions · HCC · Cholangiocarcinoma · Liver Metastases · Lymphoma · Sarcoma · Cirrhosis · Fibrosis · Steatosis · Haemochromatosis · Wilson's Disease · Liver Abscesses · Hydatid Disease · Granulomatous Diseases · Budd-Chiari Syndrome · Portal Vein Thrombosis · Traumatic Injuries · Vascular Malformations

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/akshayajay/liver-disease-diagnosis.git
cd liver-disease-diagnosis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installation varies by platform. If the above fails for `torch`, follow the [official install guide](https://pytorch.org/get-started/locally/) first.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How to Use

1. Upload a **medical image** (PNG / JPEG) — CT, MRI, or Ultrasound scan of the liver
2. Upload an **EHR text file** (TXT) — clinical notes including age, sex, imaging findings
3. Upload a **blood report** (TXT) — containing ALT, AST, ALP, and Total Bilirubin values
4. Click **Run Diagnosis**

Sample files for all three modalities are provided in the `sample/` directory.

---

## Project Structure

```
liver-disease-diagnosis/
├── app.py                    # Streamlit application
├── requirements.txt
├── .gitignore
├── README.md
├── saved_model/
│   ├── XGBclass.joblib       # Trained XGBoost classifier
│   ├── TFIDFvect.joblib      # Fitted TF-IDF vectoriser
│   └── label_remap.joblib    # Label encoder mapping
├── sample/                   # Demo patient data
│   ├── images/               # Synthetically generated medical images
│   ├── ehr/                  # Sample EHR text files
│   ├── blood_reports/        # Sample blood report text files
│   └── sample1.png / sample2.png
└── scripts/
    └── dataset.py            # Synthetic data generation (Google Colab + GPU)
```

---

## Dataset Generation

Real liver imaging data could not be procured due to privacy constraints. The training dataset was synthetically generated using a medical image diffusion model ([Prompt2MedImage](https://huggingface.co/Nihirc/Prompt2MedImage)) on Google Colab (GPU). EHR and blood report text was generated using the `Faker` library with clinically informed templates.

The generation script is in `scripts/dataset.py`. It requires a Colab environment with GPU access and Google Drive mounted.

---

## Tech Stack

`Python` · `Streamlit` · `PyTorch` · `HuggingFace Transformers (ViT)` · `XGBoost` · `scikit-learn` · `joblib`

---

## Disclaimer

This system is for **research and educational purposes only**. It is not approved for clinical use and must not be used as a substitute for diagnosis by a qualified medical professional.

---

## Author

**Akshaya J** · [github.com/akshayajay](https://github.com/akshayajay)
