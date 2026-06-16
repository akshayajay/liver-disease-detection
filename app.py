"""
Liver Disease Diagnosis and Explanation System
-----------------------------------------------
A multi-modal Streamlit application that diagnoses liver conditions from:
  - Medical imaging (CT / MRI / Ultrasound) via Vision Transformer (ViT)
  - Electronic Health Records (EHR) via TF-IDF + XGBoost
  - Blood lab values (ALT, AST, ALP, Total Bilirubin)

Run with:  streamlit run app.py
"""

import os
import re
import joblib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
import streamlit as st
import tempfile

# Resolve paths relative to this file so the app works from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")


def set_custom_page_config():
    st.set_page_config(
        page_title="Liver Diagnosis System",
        page_icon="🩺",
        layout="wide"
    )
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        body {
            background: linear-gradient(to right, #283048, #859398);
            color: #f0f0f0;
            font-family: Arial, sans-serif;
        }
        h1, h2, h3, h4 { color: #ffffff; }
        .result-box {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


class LiverDiagnosisSystem:
    """
    Multi-modal liver disease diagnosis system.

    Combines:
    - Vision Transformer (google/vit-base-patch16-224-in21k) for image features
    - TF-IDF vectoriser for EHR text features
    - Parsed blood lab values (ALT, AST, ALP, Total Bilirubin)
    - XGBoost classifier trained on the fused feature vector

    Falls back to rule-based keyword matching when saved models are unavailable.
    """

    DISEASE_TO_LABEL = {
        "Simple Hepatic Cysts": 0,
        "Hemangioma": 1,
        "Focal Nodular Hyperplasia FNH": 2,
        "Hepatic Adenoma": 3,
        "Biliary Hamartomas Von Meyenburg Complexes": 4,
        "Other Rare Benign Lesions": 5,
        "Hepatocellular Carcinoma HCC": 6,
        "Intrahepatic Cholangiocarcinoma": 7,
        "Liver Metastases": 8,
        "Lymphoma Involving the Liver": 9,
        "Primary Hepatic Sarcomas": 10,
        "Cirrhosis": 11,
        "Steatosis Fatty Liver Disease": 12,
        "Fibrosis": 13,
        "Iron Overload Disorders Hemochromatosis": 14,
        "Copper Accumulation Wilsons Disease": 15,
        "Liver Abscesses": 16,
        "Hydatid Disease": 17,
        "Granulomatous Diseases": 18,
        "Budd Chiari Syndrome": 19,
        "Portal Vein Thrombosis": 20,
        "Traumatic Injuries": 21,
        "Vascular Malformations": 22,
    }

    CLINICAL_EXPLANATIONS = {
        "Cirrhosis": (
            "Cirrhosis is characterised by diffuse liver fibrosis and regenerative nodules. "
            "Imaging typically reveals a nodular liver, splenomegaly, and signs of portal hypertension. "
            "Laboratory abnormalities include elevated bilirubin, low albumin, and prolonged prothrombin time. "
            "Management includes treating the underlying cause, managing complications, and considering "
            "liver transplantation in advanced cases."
        ),
        "Simple Hepatic Cysts": (
            "Simple hepatic cysts are benign, fluid-filled lesions that are usually asymptomatic. "
            "They appear as well-circumscribed lesions on imaging with normal laboratory tests. "
            "Treatment is generally conservative unless symptoms develop."
        ),
        "Hemangioma": (
            "Hemangiomas are benign vascular tumours that exhibit peripheral nodular enhancement "
            "with centripetal fill-in on imaging. Laboratory tests are typically normal, and most "
            "cases are managed conservatively."
        ),
        "Focal Nodular Hyperplasia FNH": (
            "Focal nodular hyperplasia (FNH) is a benign liver lesion with a characteristic "
            "central scar and radiating septa. Imaging typically shows a well-circumscribed lesion "
            "with homogeneous enhancement, and laboratory values are usually normal. "
            "Management is conservative with periodic follow-up."
        ),
        "Hepatic Adenoma": (
            "Hepatic adenomas are benign, hormone-sensitive tumours most commonly seen in young women. "
            "They may present with haemorrhage and variable imaging enhancement. "
            "Management includes surgical resection for large or symptomatic lesions."
        ),
        "Biliary Hamartomas Von Meyenburg Complexes": (
            "Biliary hamartomas (Von Meyenburg complexes) are benign malformations of the bile ducts "
            "that appear as multiple small cystic lesions on imaging. "
            "They are typically asymptomatic, and management is conservative."
        ),
        "Other Rare Benign Lesions": (
            "This category includes various rare benign liver lesions with diverse imaging and "
            "laboratory findings. Management is tailored to the lesion characteristics and patient symptoms."
        ),
        "Hepatocellular Carcinoma HCC": (
            "Hepatocellular carcinoma (HCC) is the most common primary liver malignancy, typically "
            "presenting with arterial-phase hyperenhancement and venous-phase washout on imaging. "
            "Elevated alpha-fetoprotein (AFP) is common. Treatment options include resection, ablation, "
            "and liver transplantation, depending on tumour stage and liver function."
        ),
        "Intrahepatic Cholangiocarcinoma": (
            "Intrahepatic cholangiocarcinoma is a primary bile-duct cancer within the liver, "
            "usually presenting as a mass with delayed enhancement and a cholestatic laboratory pattern. "
            "Surgical resection is the primary treatment when feasible, often combined with adjuvant chemotherapy."
        ),
        "Liver Metastases": (
            "Liver metastases are secondary tumours originating from primary cancers elsewhere in the body. "
            "They typically appear as multiple lesions with variable enhancement on imaging. "
            "Management depends on the primary cancer type and extent of metastasis."
        ),
        "Lymphoma Involving the Liver": (
            "Lymphoma involving the liver can present as diffuse infiltration or focal lesions on imaging. "
            "Diagnosis is confirmed by biopsy, and management generally involves systemic chemotherapy."
        ),
        "Primary Hepatic Sarcomas": (
            "Primary hepatic sarcomas are rare malignant tumours with heterogeneous imaging features, "
            "often requiring surgical resection followed by adjuvant therapy."
        ),
        "Steatosis Fatty Liver Disease": (
            "Fatty liver disease is characterised by the accumulation of fat in liver cells. "
            "Imaging reveals increased echogenicity on ultrasound and decreased attenuation on CT. "
            "Management focuses on lifestyle modifications such as diet and exercise."
        ),
        "Fibrosis": (
            "Liver fibrosis involves excessive deposition of extracellular matrix proteins due to "
            "chronic liver injury. Imaging may reveal a coarser liver texture, and laboratory tests "
            "often show elevated liver enzymes. Treatment is directed at the underlying cause."
        ),
        "Iron Overload Disorders Hemochromatosis": (
            "Haemochromatosis is marked by excessive iron deposition in the liver. "
            "MRI may show a darker liver signal, and laboratory tests reveal elevated serum ferritin "
            "and transferrin saturation. Management involves regular phlebotomy and iron chelation therapy."
        ),
        "Copper Accumulation Wilsons Disease": (
            "Wilson's disease is a genetic disorder resulting in abnormal copper accumulation in the "
            "liver and brain. It is characterised by low ceruloplasmin levels. "
            "Management includes copper chelation therapy and dietary modifications."
        ),
        "Liver Abscesses": (
            "Liver abscesses present as fluid-filled cavities on imaging, often accompanied by fever "
            "and leukocytosis. They are managed with antibiotic therapy and, if necessary, "
            "percutaneous drainage."
        ),
        "Hydatid Disease": (
            "Hydatid disease, caused by Echinococcus infection, produces cystic lesions in the liver "
            "with characteristic daughter cysts on imaging. "
            "Treatment includes antiparasitic medications and possible surgical intervention."
        ),
        "Granulomatous Diseases": (
            "Granulomatous liver diseases involve the formation of granulomas due to infections "
            "(e.g. tuberculosis) or inflammatory conditions (e.g. sarcoidosis). "
            "Management is tailored to the underlying cause."
        ),
        "Budd Chiari Syndrome": (
            "Budd-Chiari syndrome is caused by obstruction of hepatic venous outflow, resulting in "
            "hepatomegaly and ascites. Management includes anticoagulation and, in some cases, "
            "interventional procedures."
        ),
        "Portal Vein Thrombosis": (
            "Portal vein thrombosis is the formation of a clot in the portal vein. "
            "Doppler ultrasound or CT imaging confirms the diagnosis, and treatment primarily "
            "involves anticoagulation therapy."
        ),
        "Traumatic Injuries": (
            "Traumatic liver injuries range from minor lacerations to severe parenchymal disruptions. "
            "Imaging is used to assess the injury, and treatment varies from conservative management "
            "to surgical intervention based on severity."
        ),
        "Vascular Malformations": (
            "Vascular malformations in the liver are abnormal clusters of blood vessels, usually "
            "asymptomatic and discovered incidentally. "
            "They typically do not require treatment unless complications occur."
        ),
    }

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_to_disease = {v: k for k, v in self.DISEASE_TO_LABEL.items()}
        self.models_loaded = False
        self._initialize_models()

    def _initialize_models(self):
        xgb_path = os.path.join(MODEL_DIR, "XGBclass.joblib")
        tfidf_path = os.path.join(MODEL_DIR, "TFIDFvect.joblib")

        if not (os.path.exists(xgb_path) and os.path.exists(tfidf_path)):
            st.warning("Saved model files not found — running in rule-based fallback mode.")
            return

        try:
            self.xgb_model = joblib.load(xgb_path)
            self.tfidf_vectorizer = joblib.load(tfidf_path)
        except Exception as e:
            st.warning(f"Could not load classification models: {e}")
            return

        try:
            self.image_processor = ViTImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k"
            )
            self.vit_model = ViTModel.from_pretrained(
                "google/vit-base-patch16-224-in21k"
            ).to(self.device)
            self.vit_model.eval()
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.image_processor.image_mean,
                    std=self.image_processor.image_std,
                ),
            ])
            self.models_loaded = True
        except Exception as e:
            st.warning(f"Could not load Vision Transformer: {e}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_image_features(self, image_path: str) -> np.ndarray:
        if not self.models_loaded:
            return np.random.random(768)
        if not os.path.exists(image_path):
            return np.zeros(768)
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.vit_model(pixel_values=tensor)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        except Exception:
            return np.zeros(768)

    def extract_text_features(self, text: str) -> np.ndarray:
        if not self.models_loaded or not text:
            return np.zeros(
                self.tfidf_vectorizer.get_feature_names_out().shape[0]
                if hasattr(self, "tfidf_vectorizer")
                else 100
            )
        return self.tfidf_vectorizer.transform([text]).toarray().flatten()

    def extract_blood_features(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(4)
        features = []
        for key in ("ALT", "AST", "ALP", "Total Bilirubin"):
            match = re.search(key + r":\s*([\d\.]+)", text)
            features.append(float(match.group(1)) if match else 0.0)
        return np.array(features)

    def parse_blood_values(self, text: str) -> dict:
        values = {}
        for key in ("ALT", "AST", "ALP", "Total Bilirubin"):
            match = re.search(key + r":\s*([\d\.]+)", text or "")
            if match:
                values[key] = float(match.group(1))
        return values

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    _KEYWORD_PATTERNS = {
        "Simple Hepatic Cysts": r"\bcyst(s)?\b",
        "Hemangioma": r"\bhemangioma\b",
        "Focal Nodular Hyperplasia FNH": r"\bfnh\b|\bfocal nodular hyperplasia\b",
        "Hepatic Adenoma": r"\badenoma\b",
        "Biliary Hamartomas Von Meyenburg Complexes": r"\bhamartoma\b|\bvon meyenburg\b",
        "Other Rare Benign Lesions": r"\brare benign lesion(s)?\b",
        "Hepatocellular Carcinoma HCC": r"\bhcc\b|\bhepatocellular carcinoma\b|\bmass\b|\bnodule\b",
        "Intrahepatic Cholangiocarcinoma": r"\bcholangiocarcinoma\b",
        "Liver Metastases": r"\bmetastases?\b",
        "Lymphoma Involving the Liver": r"\blymphoma\b",
        "Primary Hepatic Sarcomas": r"\bsarcoma\b",
        "Cirrhosis": r"\bcirrhosis\b",
        "Steatosis Fatty Liver Disease": r"\bfatty liver\b|\bsteatosis\b",
        "Fibrosis": r"\bfibrosis\b",
        "Iron Overload Disorders Hemochromatosis": r"\bhemochromatosis\b|\biron overload\b",
        "Copper Accumulation Wilsons Disease": r"\bwilson('?s)? disease\b|\bcopper accumulation\b",
        "Liver Abscesses": r"\babscess(es)?\b",
        "Hydatid Disease": r"\bhydatid\b|\bechinococcus\b",
        "Granulomatous Diseases": r"\bgranulomatous\b",
        "Budd Chiari Syndrome": r"\bbudd[- ]chiari\b",
        "Portal Vein Thrombosis": r"\bportal vein thrombosis\b",
        "Traumatic Injuries": r"\btrauma\b|\binjury\b",
        "Vascular Malformations": r"\bvascular malformation(s)?\b",
    }

    _PRIORITY = [
        "Hepatocellular Carcinoma HCC", "Intrahepatic Cholangiocarcinoma",
        "Liver Metastases", "Lymphoma Involving the Liver", "Primary Hepatic Sarcomas",
        "Cirrhosis", "Fibrosis", "Steatosis Fatty Liver Disease",
        "Focal Nodular Hyperplasia FNH", "Hepatic Adenoma", "Hemangioma",
        "Biliary Hamartomas Von Meyenburg Complexes", "Other Rare Benign Lesions",
        "Iron Overload Disorders Hemochromatosis", "Copper Accumulation Wilsons Disease",
        "Liver Abscesses", "Hydatid Disease", "Granulomatous Diseases",
        "Budd Chiari Syndrome", "Portal Vein Thrombosis", "Traumatic Injuries",
        "Vascular Malformations", "Simple Hepatic Cysts",
    ]

    def _rule_based_diagnose(self, ehr_text: str, blood_values: dict) -> str:
        for disease in self._PRIORITY:
            pattern = self._KEYWORD_PATTERNS.get(disease, "")
            if pattern and re.search(pattern, ehr_text or "", re.IGNORECASE):
                return disease
        if blood_values.get("Total Bilirubin", 0) > 2.0:
            return "Cirrhosis"
        return "Simple Hepatic Cysts"

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def diagnose(self, image_path=None, ehr_text=None, blood_text=None) -> str:
        blood_values = self.parse_blood_values(blood_text or "")

        if not self.models_loaded:
            return self._rule_based_diagnose(ehr_text or "", blood_values)

        features = []
        if image_path and os.path.exists(image_path):
            features.append(self.extract_image_features(image_path))
        if ehr_text:
            features.append(self.extract_text_features(ehr_text))
        if blood_text:
            features.append(self.extract_blood_features(blood_text))

        if not features:
            return "Insufficient data for diagnosis"

        try:
            combined = np.concatenate(features).reshape(1, -1)
            pred_label = int(self.xgb_model.predict(combined)[0])
            return self.label_to_disease.get(pred_label, "Unknown Disease")
        except Exception:
            return self._rule_based_diagnose(ehr_text or "", blood_values)

    # ------------------------------------------------------------------
    # Clinical explanation
    # ------------------------------------------------------------------

    def generate_explanation(
        self,
        disease_name: str,
        patient_age=None,
        patient_sex=None,
        blood_values=None,
    ) -> str:
        explanation = self.CLINICAL_EXPLANATIONS.get(
            disease_name,
            f"No detailed explanation available for {disease_name}. "
            "Further clinical evaluation is recommended.",
        )
        if patient_age or patient_sex:
            gender = "male" if patient_sex == "M" else "female"
            prefix = f"For a {patient_age}-year-old {gender} patient, "
            explanation = prefix + explanation[0].lower() + explanation[1:]
        if blood_values and blood_values.get("Total Bilirubin", 0) > 2.0:
            explanation += (
                f" The elevated bilirubin level of {blood_values['Total Bilirubin']} "
                "indicates significant hepatocellular dysfunction."
            )
        return explanation

    # ------------------------------------------------------------------
    # Patient info helpers
    # ------------------------------------------------------------------

    def extract_patient_info(self, ehr_text: str):
        if not ehr_text:
            return None, None
        age_match = re.search(r"(?:age|years old)[:\s]+(\d+)", ehr_text, re.IGNORECASE)
        sex_match = re.search(r"(?:sex|gender)[:\s]+([MFmf])", ehr_text, re.IGNORECASE)
        age = int(age_match.group(1)) if age_match else None
        sex = sex_match.group(1).upper() if sex_match else None
        return age, sex

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_patient(self, image_path=None, ehr_path=None, blood_path=None) -> dict:
        results = {"success": False, "diagnosis": None, "explanation": None, "error": None}
        try:
            ehr_text, blood_text = None, None

            if ehr_path and os.path.exists(ehr_path):
                with open(ehr_path, "r", encoding="utf-8", errors="ignore") as f:
                    ehr_text = f.read()

            if blood_path and os.path.exists(blood_path):
                with open(blood_path, "r", encoding="utf-8", errors="ignore") as f:
                    blood_text = f.read()

            patient_age, patient_sex = self.extract_patient_info(ehr_text)
            blood_values = self.parse_blood_values(blood_text or "")

            disease_name = self.diagnose(image_path, ehr_text, blood_text)
            explanation = self.generate_explanation(
                disease_name,
                patient_age=patient_age,
                patient_sex=patient_sex,
                blood_values=blood_values,
            )
            results.update(success=True, diagnosis=disease_name, explanation=explanation)
        except Exception as e:
            results["error"] = str(e)
        return results


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def main():
    set_custom_page_config()

    st.markdown(
        """
        <div style="text-align:center; padding:1rem;
                    background-color:rgba(255,255,255,0.1); border-radius:0.5rem;">
            <h1>Liver Disease Diagnosis System</h1>
            <p>Upload patient data to receive a diagnosis and a detailed clinical summary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Upload Patient Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        image_file = st.file_uploader("Medical Image (PNG / JPEG)", type=["png", "jpeg", "jpg"])
    with col2:
        ehr_file = st.file_uploader("EHR Text File (TXT)", type=["txt"])
    with col3:
        blood_file = st.file_uploader("Blood Report (TXT)", type=["txt"])

    st.info(
        "Sample files are available in the `sample/` directory of this repository. "
        "You can use them to try the app without real patient data."
    )

    if st.button("Run Diagnosis", type="primary"):
        if not any([image_file, ehr_file, blood_file]):
            st.warning("Please upload at least one input file.")
            return

        image_path = save_uploaded_file(image_file)
        ehr_path = save_uploaded_file(ehr_file)
        blood_path = save_uploaded_file(blood_file)

        with st.spinner("Loading models and processing patient data…"):
            system = LiverDiagnosisSystem()
            results = system.process_patient(image_path, ehr_path, blood_path)

        st.header("Diagnosis Result")
        if results["success"]:
            st.success(f"**Predicted Condition:** {results['diagnosis']}")
            st.subheader("Clinical Summary")
            st.markdown(
                f"<div class='result-box'>{results['explanation']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Error: {results['error']}")
            st.write("Please check your input files and try again.")

    st.markdown("---")
    st.caption(
        "⚠️ This system is for research and educational purposes only. "
        "It is not a substitute for clinical diagnosis by a qualified medical professional."
    )


if __name__ == "__main__":
    main()
