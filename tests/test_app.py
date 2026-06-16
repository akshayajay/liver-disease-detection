"""
tests/test_app.py
-----------------
Unit tests for the LiverDiagnosisSystem logic in app.py.

All tests use synthetic inputs — no model files, no network, no GPU required.
The LiverDiagnosisSystem falls back to rule-based mode when saved models are
absent, which is exactly the behaviour we test here.
"""

import sys
import os
import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional imports before app.py is imported,
# so tests run without torch / transformers / streamlit installed.
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _mod in ("torch", "torchvision", "torchvision.transforms",
             "transformers", "streamlit", "joblib"):
    if _mod not in sys.modules:
        _make_stub(_mod)

# Minimal torch stub so `torch.cuda.is_available()` doesn't crash
torch_stub = sys.modules["torch"]
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)

# Expose the specific transformers names the app imports
tf_stub = sys.modules["transformers"]
tf_stub.ViTImageProcessor = None
tf_stub.ViTModel = None

# Minimal streamlit stub so decorators / st.warning don't crash
st_stub = sys.modules["streamlit"]
st_stub.warning = lambda *a, **k: None
st_stub.write   = lambda *a, **k: None
st_stub.error   = lambda *a, **k: None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import LiverDiagnosisSystem   # noqa: E402


# ---------------------------------------------------------------------------
# Fixture — system in rule-based fallback mode (no model files)
# ---------------------------------------------------------------------------

@pytest.fixture
def system():
    s = LiverDiagnosisSystem.__new__(LiverDiagnosisSystem)
    s.device = "cpu"
    s.models_loaded = False
    s.label_to_disease = {v: k for k, v in LiverDiagnosisSystem.DISEASE_TO_LABEL.items()}
    return s


# ---------------------------------------------------------------------------
# DISEASE_TO_LABEL sanity checks
# ---------------------------------------------------------------------------

class TestDiseaseLabelMap:
    def test_has_23_diseases(self):
        assert len(LiverDiagnosisSystem.DISEASE_TO_LABEL) == 23

    def test_labels_are_zero_indexed_contiguous(self):
        labels = sorted(LiverDiagnosisSystem.DISEASE_TO_LABEL.values())
        assert labels == list(range(23))

    def test_no_duplicate_labels(self):
        labels = list(LiverDiagnosisSystem.DISEASE_TO_LABEL.values())
        assert len(labels) == len(set(labels))

    def test_no_duplicate_disease_names(self):
        names = list(LiverDiagnosisSystem.DISEASE_TO_LABEL.keys())
        assert len(names) == len(set(names))

    def test_hcc_label(self):
        assert LiverDiagnosisSystem.DISEASE_TO_LABEL["Hepatocellular Carcinoma HCC"] == 6

    def test_cirrhosis_label(self):
        assert LiverDiagnosisSystem.DISEASE_TO_LABEL["Cirrhosis"] == 11


# ---------------------------------------------------------------------------
# parse_blood_values()
# ---------------------------------------------------------------------------

class TestParseBloodValues:
    def test_parses_all_four_markers(self, system):
        text = "ALT: 45.0\nAST: 38.5\nALP: 120.0\nTotal Bilirubin: 1.2"
        result = system.parse_blood_values(text)
        assert result == {"ALT": 45.0, "AST": 38.5, "ALP": 120.0, "Total Bilirubin": 1.2}

    def test_missing_markers_not_in_dict(self, system):
        result = system.parse_blood_values("ALT: 30.0")
        assert "ALT" in result
        assert "AST" not in result

    def test_empty_string_returns_empty_dict(self, system):
        assert system.parse_blood_values("") == {}

    def test_none_returns_empty_dict(self, system):
        assert system.parse_blood_values(None) == {}

    def test_values_are_floats(self, system):
        result = system.parse_blood_values("ALT: 55\nAST: 40")
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# extract_blood_features()
# ---------------------------------------------------------------------------

class TestExtractBloodFeatures:
    def test_returns_array_of_length_4(self, system):
        text = "ALT: 45.0\nAST: 38.5\nALP: 120.0\nTotal Bilirubin: 1.2"
        result = system.extract_blood_features(text)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4

    def test_correct_values_in_order(self, system):
        text = "ALT: 10.0\nAST: 20.0\nALP: 30.0\nTotal Bilirubin: 4.0"
        result = system.extract_blood_features(text)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0, 4.0])

    def test_missing_markers_default_to_zero(self, system):
        result = system.extract_blood_features("ALT: 55.0")
        assert result[0] == 55.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_empty_string_returns_zeros(self, system):
        result = system.extract_blood_features("")
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# extract_patient_info()
# ---------------------------------------------------------------------------

class TestExtractPatientInfo:
    def test_extracts_age_and_sex(self, system):
        ehr = "Patient: Age: 45, Sex: M\nPast history: cirrhosis."
        age, sex = system.extract_patient_info(ehr)
        assert age == 45
        assert sex == "M"

    def test_female_sex(self, system):
        ehr = "Gender: F\nAge: 30"
        age, sex = system.extract_patient_info(ehr)
        assert sex == "F"
        assert age == 30

    def test_missing_age_returns_none(self, system):
        age, _ = system.extract_patient_info("Sex: M")
        assert age is None

    def test_missing_sex_returns_none(self, system):
        _, sex = system.extract_patient_info("Age: 50")
        assert sex is None

    def test_none_input_returns_none_none(self, system):
        age, sex = system.extract_patient_info(None)
        assert age is None
        assert sex is None

    def test_sex_normalised_to_uppercase(self, system):
        _, sex = system.extract_patient_info("Gender: f")
        assert sex == "F"


# ---------------------------------------------------------------------------
# _rule_based_diagnose()
# ---------------------------------------------------------------------------

class TestRuleBasedDiagnose:
    def test_detects_cirrhosis_by_keyword(self, system):
        result = system._rule_based_diagnose("Patient has cirrhosis.", {})
        assert result == "Cirrhosis"

    def test_detects_hcc_by_keyword(self, system):
        result = system._rule_based_diagnose("Imaging shows hepatocellular carcinoma.", {})
        assert result == "Hepatocellular Carcinoma HCC"

    def test_detects_fibrosis_by_keyword(self, system):
        result = system._rule_based_diagnose("Evidence of fibrosis noted.", {})
        assert result == "Fibrosis"

    def test_detects_fatty_liver(self, system):
        result = system._rule_based_diagnose("Diagnosis: fatty liver disease.", {})
        assert result == "Steatosis Fatty Liver Disease"

    def test_high_bilirubin_suggests_cirrhosis(self, system):
        result = system._rule_based_diagnose("", {"Total Bilirubin": 3.5})
        assert result == "Cirrhosis"

    def test_normal_bilirubin_does_not_trigger_cirrhosis(self, system):
        result = system._rule_based_diagnose("", {"Total Bilirubin": 1.0})
        assert result == "Simple Hepatic Cysts"

    def test_hcc_takes_priority_over_cyst(self, system):
        result = system._rule_based_diagnose(
            "Patient has a hepatocellular carcinoma with a cyst.", {}
        )
        assert result == "Hepatocellular Carcinoma HCC"

    def test_no_input_returns_default(self, system):
        result = system._rule_based_diagnose("", {})
        assert result == "Simple Hepatic Cysts"

    def test_case_insensitive_matching(self, system):
        result = system._rule_based_diagnose("CIRRHOSIS detected.", {})
        assert result == "Cirrhosis"


# ---------------------------------------------------------------------------
# generate_explanation()
# ---------------------------------------------------------------------------

class TestGenerateExplanation:
    def test_returns_string(self, system):
        result = system.generate_explanation("Cirrhosis")
        assert isinstance(result, str)
        assert len(result) > 50

    def test_known_disease_returns_specific_text(self, system):
        result = system.generate_explanation("Cirrhosis")
        assert "fibrosis" in result.lower() or "cirrhosis" in result.lower()

    def test_unknown_disease_returns_fallback(self, system):
        result = system.generate_explanation("Mystery Condition")
        assert "Mystery Condition" in result

    def test_personalised_with_age_and_sex(self, system):
        result = system.generate_explanation("Cirrhosis", patient_age=55, patient_sex="M")
        assert "55" in result
        assert "male" in result.lower()

    def test_female_personalisation(self, system):
        result = system.generate_explanation("Cirrhosis", patient_age=40, patient_sex="F")
        assert "female" in result.lower()

    def test_high_bilirubin_appended(self, system):
        result = system.generate_explanation(
            "Cirrhosis", blood_values={"Total Bilirubin": 3.5}
        )
        assert "3.5" in result or "bilirubin" in result.lower()

    def test_normal_bilirubin_not_appended(self, system):
        result = system.generate_explanation(
            "Cirrhosis", blood_values={"Total Bilirubin": 1.0}
        )
        assert "bilirubin" not in result.lower() or "1.0" not in result

    def test_all_23_diseases_have_explanations(self, system):
        for disease in LiverDiagnosisSystem.DISEASE_TO_LABEL:
            result = system.generate_explanation(disease)
            assert isinstance(result, str) and len(result) > 20, \
                f"Missing or too-short explanation for: {disease}"
