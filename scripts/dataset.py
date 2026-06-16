import os
import random
import numpy as np
from faker import Faker
import pandas as pd
import torch
from diffusers import DiffusionPipeline
from PIL import Image

# Mount your Google Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# Initialize Faker and set seeds for reproducibility
faker = Faker()
random.seed(42)
Faker.seed(42)

# --------------------------------------------------------------------
# 1. Define the List of Liver Diseases and Imaging Modalities
# --------------------------------------------------------------------
# The diseases are defined exactly as provided.
diseases = [
    "Simple_Hepatic_Cysts",
    "Hemangioma",
    "Focal_Nodular_Hyperplasia_FNH",
    "Hepatic_Adenoma",
    "Biliary_Hamartomas_Von_Meyenburg_Complexes",
    "Other_Rare_Benign_Lesions",
    "Hepatocellular_Carcinoma_HCC",
    "Intrahepatic_Cholangiocarcinoma",
    "Liver_Metastases",
    "Lymphoma_Involving_the_Liver",
    "Primary_Hepatic_Sarcomas",
    "Cirrhosis",
    "Steatosis_Fatty_Liver_Disease",
    "Fibrosis",
    "Iron_Overload_Disorders_Hemochromatosis",
    "Copper_Accumulation_Wilsons_Disease",
    "Liver_Abscesses",
    "Hydatid_Disease",
    "Granulomatous_Diseases",
    "Budd_Chiari_Syndrome",
    "Portal_Vein_Thrombosis",
    "Traumatic_Injuries",
    "Vascular_Malformations"
]

# Imaging modalities to simulate
modalities = ["CT", "MRI", "Ultrasound"]

# Set up output directories in Google Drive
output_dir = "/content/drive/MyDrive/liver/sample"
images_dir = os.path.join(output_dir, "images")
reports_dir = os.path.join(output_dir, "reports")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# --------------------------------------------------------------------
# 2. Load the Diffusion Model
# --------------------------------------------------------------------
print("Loading Prompt2MedImage model...")
pipe = DiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
# Disable the safety checker (for internal research only)
if hasattr(pipe, "safety_checker"):
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

# Define a negative prompt to discourage unwanted artistic artifacts
negative_prompt_text = (
    "cartoon, painting, drawing, bright neon colors, abstract, text, watermark, signature, "
    "colorful, fantasy, unreal, symmetrical pattern, glitch, fake, unrealistic"
)

# --------------------------------------------------------------------
# 3. Baseline Image Loading and Vectorisation
# --------------------------------------------------------------------
def vectorise_image(image):
    """
    A simple function to simulate vectorisation of an image.
    In practice, you might use a pretrained encoder (e.g. CLIP) to generate a latent vector.
    """
    image_gray = image.convert("L").resize((64, 64))
    vector = np.array(image_gray).flatten()
    return vector

def load_baseline_images():
    """
    Load healthy liver images for each modality from a baseline folder.
    This function scans all files in the specified directory and picks the first image file
    that has the modality keyword in its name (case insensitive).
    It creates and returns a dictionary of images and their vector representations.
    """
    baseline_dir = "/content/drive/MyDrive/liver/healthy_liver"  # Folder with healthy liver images
    baseline_images = {}
    baseline_vectors = {}

    try:
        files = os.listdir(baseline_dir)
    except Exception as e:
        print(f"Error accessing directory {baseline_dir}: {e}")
        return baseline_images, baseline_vectors

    for modality in modalities:
        modality_lower = modality.lower()
        candidate_file = None
        # Search for any file that includes the modality keyword.
        for fname in files:
            if modality_lower in fname.lower():
                candidate_file = os.path.join(baseline_dir, fname)
                break
        if candidate_file is not None and os.path.exists(candidate_file):
            image = Image.open(candidate_file).convert("RGB")
            baseline_images[modality] = image
            baseline_vectors[modality] = vectorise_image(image)
            print(f"Loaded baseline image for {modality}: {candidate_file}")
        else:
            print(f"Warning: No baseline image found for {modality}. {modality} generation will be text-driven only.")
            baseline_images[modality] = None
            baseline_vectors[modality] = None

    return baseline_images, baseline_vectors

baseline_images, baseline_vectors = load_baseline_images()

# --------------------------------------------------------------------
# 4. Create Detailed, Modality-Specific Prompts for Each Disease
# --------------------------------------------------------------------
def create_prompt(disease, modality):
    """
    Construct a detailed, modality-specific prompt for the given liver disease.
    Each prompt emphasizes a full cross-sectional view and realistic radiologic features.
    """
    # For CT modality
    if modality == "CT":
        if disease == "Simple_Hepatic_Cysts":
            return ("A high-resolution grayscale CT scan of a human liver showing a simple hepatic cyst. "
                    "The full cross-sectional image displays a well-defined, fluid-filled cyst with thin walls and no septations.")
        elif disease == "Hemangioma":
            return ("A detailed grayscale CT scan of a human liver with a hemangioma. "
                    "The full cross-sectional view demonstrates a well-circumscribed lesion with peripheral nodular enhancement and gradual fill-in.")
        elif disease == "Focal_Nodular_Hyperplasia_FNH":
            return ("A high-resolution grayscale CT scan of a human liver exhibiting focal nodular hyperplasia (FNH). "
                    "The image shows a central scar with radial vessels and a well-demarcated, non-capsulated lesion on the full cross-sectional view.")
        elif disease == "Hepatic_Adenoma":
            return ("A detailed grayscale CT scan of a human liver with a hepatic adenoma. "
                    "The full cross-sectional image shows a solitary, well-marginated lesion with heterogeneous enhancement and subtle hemorrhagic areas.")
        elif disease == "Biliary_Hamartomas_Von_Meyenburg_Complexes":
            return ("A high-resolution grayscale CT scan of a human liver with biliary hamartomas (Von Meyenburg Complexes). "
                    "The full cross-sectional view reveals multiple small, hypodense lesions with irregular margins scattered throughout the liver parenchyma.")
        elif disease == "Other_Rare_Benign_Lesions":
            return ("A detailed grayscale CT scan of a human liver displaying other rare benign lesions. "
                    "The full cross-sectional image shows small, well-defined lesions with variable enhancement patterns and clear anatomical detail.")
        elif disease == "Hepatocellular_Carcinoma_HCC":
            return ("A high-resolution grayscale CT scan of a human liver with hepatocellular carcinoma (HCC). "
                    "The full cross-sectional view exhibits a dominant mass with arterial phase hyperenhancement and washout in later phases, with irregular margins.")
        elif disease == "Intrahepatic_Cholangiocarcinoma":
            return ("A detailed grayscale CT scan of a human liver with intrahepatic cholangiocarcinoma. "
                    "The full cross-sectional image demonstrates an irregular mass with delayed contrast enhancement and peripheral rim enhancement.")
        elif disease == "Liver_Metastases":
            return ("A high-resolution grayscale CT scan of a human liver with liver metastases. "
                    "The full cross-sectional view reveals multiple hypodense lesions of varying sizes scattered in the liver parenchyma, suggestive of metastatic disease.")
        elif disease == "Lymphoma_Involving_the_Liver":
            return ("A detailed grayscale CT scan of a human liver with lymphoma involvement. "
                    "The full cross-sectional image shows diffuse infiltration with multiple ill-defined hypodense areas, without clear margins.")
        elif disease == "Primary_Hepatic_Sarcomas":
            return ("A high-resolution grayscale CT scan of a human liver with primary hepatic sarcoma. "
                    "The full cross-sectional view displays a large, heterogeneous mass with irregular borders and areas of necrosis.")
        elif disease == "Cirrhosis":
            return ("A detailed grayscale CT scan of a human liver with cirrhosis. "
                    "The full cross-sectional image demonstrates a nodular liver surface with regenerative nodules, heterogeneous parenchymal density, and distorted vascular structures.")
        elif disease == "Steatosis_Fatty_Liver_Disease":
            return ("A high-resolution grayscale CT scan of a human liver with steatosis (fatty liver disease). "
                    "The full cross-sectional view shows diffusely decreased liver attenuation with a uniformly bright parenchyma indicating fatty infiltration.")
        elif disease == "Fibrosis":
            return ("A detailed grayscale CT scan of a human liver with fibrosis. "
                    "The full cross-sectional image demonstrates irregular contours, mild nodularity, and areas of increased density due to fibrotic bands.")
        elif disease == "Iron_Overload_Disorders_Hemochromatosis":
            return ("A high-resolution grayscale CT scan of a human liver with iron overload (hemochromatosis). "
                    "The full cross-sectional view reveals a diffusely hyperdense liver parenchyma with subtle inhomogeneities caused by iron deposition.")
        elif disease == "Copper_Accumulation_Wilsons_Disease":
            return ("A detailed grayscale CT scan of a human liver with Wilson's disease. "
                    "The full cross-sectional image shows heterogeneous parenchymal density with early cirrhotic changes and subtle signs of copper accumulation.")
        elif disease == "Liver_Abscesses":
            return ("A high-resolution grayscale CT scan of a human liver with liver abscesses. "
                    "The full cross-sectional view demonstrates one or more round, hypodense lesions with a rim of enhancement and surrounding inflammatory changes.")
        elif disease == "Hydatid_Disease":
            return ("A detailed grayscale CT scan of a human liver with hydatid disease. "
                    "The full cross-sectional image shows cystic lesions with internal septations, calcifications, and possible daughter cysts, typical of echinococcal infection.")
        elif disease == "Granulomatous_Diseases":
            return ("A high-resolution grayscale CT scan of a human liver with granulomatous diseases. "
                    "The full cross-sectional view reveals multiple small, ill-defined hypodense areas with possible calcifications, suggesting a granulomatous process.")
        elif disease == "Budd_Chiari_Syndrome":
            return ("A detailed grayscale CT scan of a human liver with Budd-Chiari syndrome. "
                    "The full cross-sectional image shows an enlarged liver with heterogeneous enhancement and occlusion or narrowing of hepatic veins.")
        elif disease == "Portal_Vein_Thrombosis":
            return ("A high-resolution grayscale CT scan of a human liver with portal vein thrombosis. "
                    "The full cross-sectional view demonstrates a lack of contrast in the portal vein with secondary signs of hepatic congestion.")
        elif disease == "Traumatic_Injuries":
            return ("A detailed grayscale CT scan of a human liver with traumatic injuries. "
                    "The full cross-sectional image shows areas of laceration, irregular parenchymal disruption, and possible hematoma formation.")
        elif disease == "Vascular_Malformations":
            return ("A high-resolution grayscale CT scan of a human liver with vascular malformations. "
                    "The full cross-sectional view reveals abnormally dilated vascular channels and a disorganized vascular network within the liver parenchyma.")
        else:
            return ("A high resolution grayscale CT scan of a human liver showing signs of " + disease +
                    ", with full anatomical detail and realistic radiologic artifacts.")
    # For MRI modality (using T2-weighted sequences)
    elif modality == "MRI":
        if disease == "Simple_Hepatic_Cysts":
            return ("A high-resolution T2-weighted MRI scan of a human liver showing a simple hepatic cyst. "
                    "The full cross-sectional image exhibits a bright, homogeneous lesion with well-defined margins and no internal septations.")
        elif disease == "Hemangioma":
            return ("A detailed T2-weighted MRI scan of a human liver with a hemangioma. "
                    "The full cross-sectional view demonstrates a lesion with high signal intensity centrally and peripheral nodular enhancement.")
        elif disease == "Focal_Nodular_Hyperplasia_FNH":
            return ("A high-resolution T2-weighted MRI scan of a human liver exhibiting focal nodular hyperplasia (FNH). "
                    "The image shows a central stellate scar with radiating fibrous septa and a well-demarcated lesion on the full cross-sectional view.")
        elif disease == "Hepatic_Adenoma":
            return ("A detailed T2-weighted MRI scan of a human liver with a hepatic adenoma. "
                    "The full cross-sectional image displays a solitary lesion with heterogeneous signal intensity and subtle internal hemorrhage.")
        elif disease == "Biliary_Hamartomas_Von_Meyenburg_Complexes":
            return ("A high-resolution T2-weighted MRI scan of a human liver with biliary hamartomas (Von Meyenburg Complexes). "
                    "The full cross-sectional view shows multiple tiny cystic lesions with high signal intensity scattered throughout the liver.")
        elif disease == "Other_Rare_Benign_Lesions":
            return ("A detailed T2-weighted MRI scan of a human liver with other rare benign lesions. "
                    "The full cross-sectional image reveals small, well-defined lesions with variable signal intensities and clear margins.")
        elif disease == "Hepatocellular_Carcinoma_HCC":
            return ("A high-resolution T2-weighted MRI scan of a human liver with hepatocellular carcinoma (HCC). "
                    "The full cross-sectional view demonstrates a dominant lesion with heterogeneous signal intensity, a central necrotic core, and irregular borders.")
        elif disease == "Intrahepatic_Cholangiocarcinoma":
            return ("A detailed T2-weighted MRI scan of a human liver with intrahepatic cholangiocarcinoma. "
                    "The full cross-sectional image shows an irregular mass with peripheral high signal intensity and delayed enhancement patterns.")
        elif disease == "Liver_Metastases":
            return ("A high-resolution T2-weighted MRI scan of a human liver with liver metastases. "
                    "The full cross-sectional view reveals multiple lesions with variable sizes and heterogeneous signal intensities spread throughout the liver.")
        elif disease == "Lymphoma_Involving_the_Liver":
            return ("A detailed T2-weighted MRI scan of a human liver with lymphoma involvement. "
                    "The full cross-sectional image shows diffuse or multifocal areas of altered signal intensity without a well-defined mass.")
        elif disease == "Primary_Hepatic_Sarcomas":
            return ("A high-resolution T2-weighted MRI scan of a human liver with primary hepatic sarcoma. "
                    "The full cross-sectional view displays a large, heterogeneous mass with irregular borders and central necrosis.")
        elif disease == "Cirrhosis":
            return ("A detailed T2-weighted MRI scan of a human liver with cirrhosis. "
                    "The full cross-sectional image demonstrates a nodular liver surface with areas of fibrosis and regenerative nodules, along with heterogeneous signal intensities.")
        elif disease == "Steatosis_Fatty_Liver_Disease":
            return ("A high-resolution T2-weighted MRI scan of a human liver with steatosis (fatty liver disease). "
                    "The full cross-sectional view shows diffuse fat infiltration with overall reduced signal intensity in the liver parenchyma.")
        elif disease == "Fibrosis":
            return ("A detailed T2-weighted MRI scan of a human liver with fibrosis. "
                    "The full cross-sectional image reveals patchy areas of low signal intensity corresponding to fibrotic bands and architectural distortion.")
        elif disease == "Iron_Overload_Disorders_Hemochromatosis":
            return ("A high-resolution T2-weighted MRI scan of a human liver with iron overload (hemochromatosis). "
                    "The full cross-sectional view demonstrates a diffusely darkened liver parenchyma with inhomogeneous signal due to iron deposition.")
        elif disease == "Copper_Accumulation_Wilsons_Disease":
            return ("A detailed T2-weighted MRI scan of a human liver with Wilson's disease. "
                    "The full cross-sectional image shows heterogeneous signal intensity with subtle cirrhotic changes and indications of copper accumulation.")
        elif disease == "Liver_Abscesses":
            return ("A high-resolution T2-weighted MRI scan of a human liver with liver abscesses. "
                    "The full cross-sectional view demonstrates one or more rounded lesions with high signal intensity cores and surrounding edema.")
        elif disease == "Hydatid_Disease":
            return ("A detailed T2-weighted MRI scan of a human liver with hydatid disease. "
                    "The full cross-sectional image shows cystic lesions with high signal intensity, internal septations, and possible daughter cysts.")
        elif disease == "Granulomatous_Diseases":
            return ("A high-resolution T2-weighted MRI scan of a human liver with granulomatous diseases. "
                    "The full cross-sectional view reveals multiple small foci of altered signal intensity, some with calcifications, suggesting granuloma formation.")
        elif disease == "Budd_Chiari_Syndrome":
            return ("A detailed T2-weighted MRI scan of a human liver with Budd-Chiari syndrome. "
                    "The full cross-sectional image demonstrates an enlarged liver with heterogeneous signal intensity and evidence of hepatic vein occlusion.")
        elif disease == "Portal_Vein_Thrombosis":
            return ("A high-resolution T2-weighted MRI scan of a human liver with portal vein thrombosis. "
                    "The full cross-sectional view shows an absence of normal flow voids in the portal vein along with secondary signs of congestion in the liver.")
        elif disease == "Traumatic_Injuries":
            return ("A detailed T2-weighted MRI scan of a human liver with traumatic injuries. "
                    "The full cross-sectional image shows areas of parenchymal disruption, edema, and possible hemorrhage resulting from trauma.")
        elif disease == "Vascular_Malformations":
            return ("A high-resolution T2-weighted MRI scan of a human liver with vascular malformations. "
                    "The full cross-sectional view reveals an abnormal, disorganized network of vessels with areas of high and low signal intensity.")
        else:
            return ("A high resolution T2-weighted MRI scan of a human liver showing signs of " + disease +
                    ", with full anatomical detail and realistic radiologic features.")
    # For Ultrasound modality
    elif modality == "Ultrasound":
        if disease == "Simple_Hepatic_Cysts":
            return ("A high-resolution grayscale ultrasound image of a human liver showing a simple hepatic cyst. "
                    "The full-view image displays an anechoic, well-circumscribed cyst with posterior acoustic enhancement.")
        elif disease == "Hemangioma":
            return ("A detailed grayscale ultrasound image of a human liver with a hemangioma. "
                    "The full-view image demonstrates a hyperechoic lesion with smooth, well-defined margins and possible posterior acoustic shadowing.")
        elif disease == "Focal_Nodular_Hyperplasia_FNH":
            return ("A high-resolution grayscale ultrasound image of a human liver with focal nodular hyperplasia (FNH). "
                    "The full-view image shows a well-demarcated lesion with a central stellate scar and radiating vessels.")
        elif disease == "Hepatic_Adenoma":
            return ("A detailed grayscale ultrasound image of a human liver with hepatic adenoma. "
                    "The full-view image reveals a solitary, hypoechoic to isoechoic lesion with irregular internal texture.")
        elif disease == "Biliary_Hamartomas_Von_Meyenburg_Complexes":
            return ("A high-resolution grayscale ultrasound image of a human liver with biliary hamartomas (Von Meyenburg Complexes). "
                    "The full-view image shows multiple small, hypoechoic lesions scattered throughout the liver.")
        elif disease == "Other_Rare_Benign_Lesions":
            return ("A detailed grayscale ultrasound image of a human liver with other rare benign lesions. "
                    "The full-view image displays small, round lesions with variable echotexture and clear boundaries.")
        elif disease == "Hepatocellular_Carcinoma_HCC":
            return ("A high-resolution grayscale ultrasound image of a human liver with hepatocellular carcinoma (HCC). "
                    "The full-view image exhibits a dominant, heterogeneous lesion with irregular margins and internal vascularity.")
        elif disease == "Intrahepatic_Cholangiocarcinoma":
            return ("A detailed grayscale ultrasound image of a human liver with intrahepatic cholangiocarcinoma. "
                    "The full-view image shows an irregular mass with heterogeneous echogenicity and peripheral vascularity.")
        elif disease == "Liver_Metastases":
            return ("A high-resolution grayscale ultrasound image of a human liver with liver metastases. "
                    "The full-view image demonstrates multiple hypoechoic lesions of varying sizes distributed throughout the liver parenchyma.")
        elif disease == "Lymphoma_Involving_the_Liver":
            return ("A detailed grayscale ultrasound image of a human liver with lymphoma involvement. "
                    "The full-view image reveals diffuse heterogeneity with ill-defined hypoechoic areas without a discrete mass.")
        elif disease == "Primary_Hepatic_Sarcomas":
            return ("A high-resolution grayscale ultrasound image of a human liver with primary hepatic sarcoma. "
                    "The full-view image displays a large, irregular, heterogeneous mass with regions of necrosis and calcification.")
        elif disease == "Cirrhosis":
            return ("A detailed grayscale ultrasound image of a human liver with cirrhosis. "
                    "The full-view image shows a nodular, shrunken liver with coarse echotexture, irregular margins, and signs of regenerative nodules.")
        elif disease == "Steatosis_Fatty_Liver_Disease":
            return ("A high-resolution grayscale ultrasound image of a human liver with steatosis (fatty liver disease). "
                    "The full-view image demonstrates an increased echogenicity of the liver parenchyma with a bright, homogeneous appearance.")
        elif disease == "Fibrosis":
            return ("A detailed grayscale ultrasound image of a human liver with fibrosis. "
                    "The full-view image reveals a heterogeneous liver texture with patchy areas of increased echogenicity corresponding to fibrotic bands.")
        elif disease == "Iron_Overload_Disorders_Hemochromatosis":
            return ("A high-resolution grayscale ultrasound image of a human liver with iron overload (hemochromatosis). "
                    "The full-view image shows a diffusely hyperechoic liver with subtle heterogeneous texture due to iron deposition.")
        elif disease == "Copper_Accumulation_Wilsons_Disease":
            return ("A detailed grayscale ultrasound image of a human liver with Wilson's disease. "
                    "The full-view image demonstrates subtle heterogeneous echotexture with early signs of cirrhotic changes.")
        elif disease == "Liver_Abscesses":
            return ("A high-resolution grayscale ultrasound image of a human liver with liver abscesses. "
                    "The full-view image reveals one or more hypoechoic lesions with irregular, thickened walls and surrounding hyperechoic inflammatory tissue.")
        elif disease == "Hydatid_Disease":
            return ("A detailed grayscale ultrasound image of a human liver with hydatid disease. "
                    "The full-view image shows cystic lesions with internal septations, wall calcifications, and daughter cysts causing complex echogenic patterns.")
        elif disease == "Granulomatous_Diseases":
            return ("A high-resolution grayscale ultrasound image of a human liver with granulomatous diseases. "
                    "The full-view image displays multiple small, ill-defined hypoechoic areas with occasional calcifications throughout the liver parenchyma.")
        elif disease == "Budd_Chiari_Syndrome":
            return ("A detailed grayscale ultrasound image of a human liver with Budd-Chiari syndrome. "
                    "The full-view image shows an enlarged liver with heterogeneous echotexture and absence or reduction of normal venous flow in the hepatic veins.")
        elif disease == "Portal_Vein_Thrombosis":
            return ("A high-resolution grayscale ultrasound image of a human liver with portal vein thrombosis. "
                    "The full-view image demonstrates a loss of normal anechoic flow in the portal vein with surrounding signs of congestion.")
        elif disease == "Traumatic_Injuries":
            return ("A detailed grayscale ultrasound image of a human liver with traumatic injuries. "
                    "The full-view image shows irregular liver contours, disrupted parenchymal texture, and possible fluid collections indicating hematoma.")
        elif disease == "Vascular_Malformations":
            return ("A high-resolution grayscale ultrasound image of a human liver with vascular malformations. "
                    "The full-view image reveals an abnormal network of vessels with variable echogenicity and chaotic flow patterns.")
        else:
            return ("A high resolution grayscale ultrasound image of a human liver showing signs of " + disease +
                    ", with full anatomical detail and realistic sonographic features.")
    else:
        return ("A realistic medical image of a human liver with signs of " + disease + " showing full anatomical detail.")

# --------------------------------------------------------------------
# 5. Generate Medical Image Using the Detailed Prompt and Baseline Vector
# --------------------------------------------------------------------
def generate_medical_image(disease, modality, baseline_img, width=512, height=512):
    """
    Generate a medical image using the Prompt2MedImage model with a detailed prompt.
    When a healthy baseline image is available it is used as the initialization image.
    """
    prompt = create_prompt(disease, modality)
    print(f"\n[Generating {modality} image for {disease}]\nPrompt:\n{prompt}\n")
    
    if baseline_img is not None:
        print("Using healthy baseline image as reference for conditioning.")
        result = pipe(
            prompt,
            init_image=baseline_img,
            strength=0.8,
            negative_prompt=negative_prompt_text,
            height=height,
            width=width,
            num_inference_steps=75,
            guidance_scale=15.0
        )
    else:
        result = pipe(
            prompt,
            negative_prompt=negative_prompt_text,
            height=height,
            width=width,
            num_inference_steps=75,
            guidance_scale=15.0
        )
    image = result.images[0]
    return image

# --------------------------------------------------------------------
# 6. Synthetic EHR and Blood Report Generation Functions
# --------------------------------------------------------------------
def generate_synthetic_ehr(patient_id, disease):
    """
    Generate a synthetic EHR record with fake patient details and a clinical narrative.
    """
    name = faker.name()
    age = random.randint(18, 90)
    gender = random.choice(["Male", "Female"])
    report = (f"Patient {name} (ID: {patient_id}), a {age}-year-old {gender}, presents with clinical signs of {disease}. "
              "Diagnostic imaging and laboratory tests are recommended for further evaluation.")
    return {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "disease": disease,
        "ehr_report": report
    }

def generate_synthetic_blood_report(patient_id, disease):
    """
    Generate a synthetic blood report that simulates abnormal lab values based on the specific liver disease.
    """
    # Base (healthy) lab values
    lab_values = {
        "ALT": random.randint(20, 40),
        "AST": random.randint(20, 40),
        "ALP": random.randint(80, 120),
        "Total_Bilirubin": round(random.uniform(0.5, 1.0), 1),
        "Albumin": round(random.uniform(3.5, 5.0), 1),
        "Prothrombin_Time": round(random.uniform(11, 13), 1)
    }
    # Disease-specific modifications
    if "Cirrhosis" in disease:
        lab_values["ALT"] += random.randint(20, 40)
        lab_values["AST"] += random.randint(30, 50)
        lab_values["ALP"] += random.randint(20, 40)
        lab_values["Total_Bilirubin"] += round(random.uniform(1.0, 2.5), 1)
        lab_values["Albumin"] -= round(random.uniform(0.5, 1.5), 1)
    elif "Hepatocellular_Carcinoma_HCC" in disease or "Cholangiocarcinoma" in disease:
        lab_values["ALT"] += random.randint(50, 100)
        lab_values["AST"] += random.randint(30, 80)
        lab_values["Total_Bilirubin"] += round(random.uniform(0.5, 1.5), 1)
    elif "Hemochromatosis" in disease:
        lab_values["ALT"] += random.randint(10, 30)
        lab_values["AST"] += random.randint(10, 30)
    # Additional modifications for other diseases can be added as needed.
    report = (f"Blood Report for Patient {patient_id} with {disease}:\n"
              f"- ALT (Alanine Transaminase): {lab_values['ALT']} U/L\n"
              f"- AST (Aspartate Transaminase): {lab_values['AST']} U/L\n"
              f"- ALP (Alkaline Phosphatase): {lab_values['ALP']} U/L\n"
              f"- Total Bilirubin: {lab_values['Total_Bilirubin']} mg/dL\n"
              f"- Albumin: {lab_values['Albumin']} g/dL\n"
              f"- Prothrombin Time: {lab_values['Prothrombin_Time']} seconds\n")
    return report

# --------------------------------------------------------------------
# 7. Synthetic Image Generation and Dataset Assembly Functions
# --------------------------------------------------------------------
def generate_synthetic_image(patient_id, disease, modality, baseline_images):
    """
    Generate and save a synthetic medical image for the given disease and modality.
    The healthy baseline image is used as conditioning if available.
    """
    baseline_img = baseline_images.get(modality)
    img = generate_medical_image(disease, modality, baseline_img, width=512, height=512)
    sanitized_disease = disease.replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
    filename = f"{patient_id}_{modality}_{sanitized_disease}.png"
    filepath = os.path.join(images_dir, filename)
    img.save(filepath)
    return filepath

def save_txt_report(patient_id, report_text, report_type="ehr"):
    """
    Save a text report (EHR or blood report) to a .txt file.
    """
    filename = f"{patient_id}_{report_type}.txt"
    filepath = os.path.join(reports_dir, filename)
    with open(filepath, "w") as f:
        f.write(report_text)
    return filepath

def generate_dataset():
    """
    Generate a synthetic multi-modal dataset covering all provided liver diseases.
    For each disease, create a synthetic EHR, a synthetic blood report, and generate an image for each modality.
    Save the reports as .txt files and record all file paths into a CSV file.
    """
    records = []
    for disease in diseases:
        pid = "PID_" + disease
        # Generate synthetic EHR and blood report
        ehr_record = generate_synthetic_ehr(pid, disease)
        blood_report = generate_synthetic_blood_report(pid, disease)
        
        # Save EHR and blood report as txt files
        ehr_txt_path = save_txt_report(pid, ehr_record["ehr_report"], report_type="ehr")
        blood_txt_path = save_txt_report(pid, blood_report, report_type="blood_report")
        ehr_record["ehr_txt"] = ehr_txt_path
        ehr_record["blood_report_txt"] = blood_txt_path
        
        # Generate images for each modality using the healthy baseline as reference
        for modality in modalities:
            img_path = generate_synthetic_image(pid, disease, modality, baseline_images)
            ehr_record[f"{modality}_image"] = img_path
        
        records.append(ehr_record)
    
    # Save records into a CSV file
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "synthetic_ehr_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataset successfully saved to: {csv_path}")
    return records

# --------------------------------------------------------------------
# 8. Main Execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    dataset = generate_dataset()
