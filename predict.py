"""
Crop Disease Detection — Inference Script
Usage: python predict.py --image path/to/leaf.jpg
"""

import argparse
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

IMG_SIZE      = 224
DENSE_UNITS   = 512
DROPOUT       = 0.5
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
#  Disease information database
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "status": "DISEASED",
        "description": "A fungal disease causing dark, scabby lesions on leaves and fruit.",
        "cause": "Fungus: Venturia inaequalis",
        "symptoms": "Olive-green to brown scabby spots on leaves and fruit surface.",
        "treatment": "Apply fungicides (captan or myclobutanil). Remove infected leaves. Improve air circulation.",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "status": "DISEASED",
        "description": "A fungal disease that causes leaf spots, fruit rot, and cankers on branches.",
        "cause": "Fungus: Botryosphaeria obtusa",
        "symptoms": "Purple spots on leaves, brown to black rotting fruit, sunken cankers on bark.",
        "treatment": "Prune infected branches, apply copper-based fungicides, remove mummified fruit.",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "status": "DISEASED",
        "description": "A fungal disease requiring both apple and cedar/juniper trees to complete its life cycle.",
        "cause": "Fungus: Gymnosporangium juniperi-virginianae",
        "symptoms": "Bright orange-yellow spots on upper leaf surface, tube-like structures underneath.",
        "treatment": "Apply fungicides in spring, remove nearby juniper/cedar trees if possible.",
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The apple plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular watering, fertilizing, and monitoring.",
        "severity": "None"
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The blueberry plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "status": "DISEASED",
        "description": "A fungal disease that creates a white powdery coating on plant surfaces.",
        "cause": "Fungus: Podosphaera clandestina",
        "symptoms": "White powdery patches on leaves, stunted growth, distorted new shoots.",
        "treatment": "Apply sulfur-based fungicides, improve air circulation, avoid overhead watering.",
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The cherry plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "status": "DISEASED",
        "description": "A serious fungal disease that can significantly reduce corn yields.",
        "cause": "Fungus: Cercospora zeae-maydis",
        "symptoms": "Rectangular gray to tan lesions running parallel to leaf veins.",
        "treatment": "Use resistant varieties, apply fungicides (strobilurins), practice crop rotation.",
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)",
        "disease": "Common Rust",
        "status": "DISEASED",
        "description": "A fungal disease producing rust-colored pustules on corn leaves.",
        "cause": "Fungus: Puccinia sorghi",
        "symptoms": "Small, oval, brick-red to brown pustules scattered on both leaf surfaces.",
        "treatment": "Plant resistant hybrids, apply fungicides early if severe infection occurs.",
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "status": "DISEASED",
        "description": "A fungal disease causing large lesions that can devastate corn crops.",
        "cause": "Fungus: Exserohilum turcicum",
        "symptoms": "Large, cigar-shaped gray-green to tan lesions on leaves.",
        "treatment": "Use resistant varieties, apply fungicides, practice crop rotation.",
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The corn plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "status": "DISEASED",
        "description": "One of the most destructive fungal diseases of grapes.",
        "cause": "Fungus: Guignardia bidwellii",
        "symptoms": "Brown circular lesions on leaves with black borders, shriveled black fruit.",
        "treatment": "Apply fungicides (mancozeb or myclobutanil), remove infected material, prune for airflow.",
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "status": "DISEASED",
        "description": "A complex fungal disease affecting the wood and leaves of grapevines.",
        "cause": "Multiple fungi including Phaeomoniella chlamydospora",
        "symptoms": "Tiger-stripe pattern on leaves, dark spots on berries, internal wood discoloration.",
        "treatment": "No complete cure. Remove infected wood, apply wound protectants, manage vine stress.",
        "severity": "High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "status": "DISEASED",
        "description": "A fungal leaf disease that causes defoliation in severe cases.",
        "cause": "Fungus: Isariopsis clavispora",
        "symptoms": "Irregular dark brown spots with yellow halos on leaves, premature leaf drop.",
        "treatment": "Apply copper-based fungicides, improve air circulation, remove infected leaves.",
        "severity": "Moderate"
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The grape plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "status": "DISEASED",
        "description": "One of the most devastating citrus diseases worldwide — no cure exists.",
        "cause": "Bacteria: Candidatus Liberibacter spp. spread by Asian citrus psyllid",
        "symptoms": "Yellow shoots, blotchy mottled leaves, small lopsided bitter fruit.",
        "treatment": "No cure. Remove infected trees, control psyllid insects, use disease-free nursery stock.",
        "severity": "Critical"
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "status": "DISEASED",
        "description": "A bacterial disease causing significant damage to peach leaves and fruit.",
        "cause": "Bacteria: Xanthomonas arboricola pv. pruni",
        "symptoms": "Small water-soaked spots on leaves turning brown, shot-hole appearance, fruit lesions.",
        "treatment": "Apply copper bactericides, use resistant varieties, avoid overhead irrigation.",
        "severity": "Moderate"
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The peach plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "status": "DISEASED",
        "description": "A bacterial disease that affects leaves, stems, and fruit of pepper plants.",
        "cause": "Bacteria: Xanthomonas euvesicatoria",
        "symptoms": "Small water-soaked lesions turning brown with yellow halos, raised scabs on fruit.",
        "treatment": "Apply copper bactericides, use certified disease-free seeds, practice crop rotation.",
        "severity": "Moderate"
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The bell pepper plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "status": "DISEASED",
        "description": "A common fungal disease that affects potato leaves and tubers.",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings (target board pattern) on older leaves.",
        "treatment": "Apply fungicides (chlorothalonil), remove infected leaves, ensure proper nutrition.",
        "severity": "Moderate"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "status": "DISEASED",
        "description": "The same disease that caused the Irish Potato Famine — extremely destructive.",
        "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Water-soaked dark lesions on leaves, white mold on undersides, rapid plant collapse.",
        "treatment": "Apply fungicides (metalaxyl), destroy infected plants immediately, use resistant varieties.",
        "severity": "Critical"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The potato plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The raspberry plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The soybean plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "status": "DISEASED",
        "description": "A very common fungal disease of squash and other cucurbits.",
        "cause": "Fungus: Podosphaera xanthii",
        "symptoms": "White powdery coating on leaves and stems, yellowing and wilting of leaves.",
        "treatment": "Apply potassium bicarbonate or sulfur fungicide, improve air circulation.",
        "severity": "Moderate"
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "status": "DISEASED",
        "description": "A fungal disease that causes leaf tissue to die, resembling scorch damage.",
        "cause": "Fungus: Diplocarpon earlianum",
        "symptoms": "Small purple to red spots on leaves expanding with gray centers, leaf edges browning.",
        "treatment": "Apply fungicides (captan), remove infected leaves, avoid overhead watering.",
        "severity": "Moderate"
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The strawberry plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "status": "DISEASED",
        "description": "A bacterial disease causing significant losses in tomato production.",
        "cause": "Bacteria: Xanthomonas vesicatoria",
        "symptoms": "Small water-soaked spots on leaves, raised scabs on fruit, defoliation.",
        "treatment": "Apply copper bactericides, use disease-free seeds, practice crop rotation.",
        "severity": "Moderate"
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "status": "DISEASED",
        "description": "A very common fungal disease affecting tomato leaves, stems, and fruit.",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings on lower leaves, yellow halo around spots.",
        "treatment": "Apply fungicides (chlorothalonil or mancozeb), remove lower infected leaves.",
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "status": "DISEASED",
        "description": "An extremely destructive disease that can destroy entire tomato crops rapidly.",
        "cause": "Oomycete: Phytophthora infestans",
        "symptoms": "Greasy gray-green water-soaked lesions, white mold on leaf undersides, brown fruit.",
        "treatment": "Apply fungicides (metalaxyl or chlorothalonil) preventively, remove infected plants.",
        "severity": "Critical"
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "status": "DISEASED",
        "description": "A fungal disease most common in greenhouses and humid conditions.",
        "cause": "Fungus: Passalora fulva",
        "symptoms": "Pale green or yellow spots on upper leaf surface, olive-green mold on undersides.",
        "treatment": "Improve ventilation, reduce humidity, apply fungicides (chlorothalonil).",
        "severity": "Moderate"
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "status": "DISEASED",
        "description": "One of the most common and destructive diseases of tomato foliage.",
        "cause": "Fungus: Septoria lycopersici",
        "symptoms": "Numerous small circular spots with dark borders and lighter centers on lower leaves.",
        "treatment": "Apply fungicides (mancozeb or chlorothalonil), remove infected leaves, mulch soil.",
        "severity": "Moderate"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-spotted)",
        "status": "DISEASED",
        "description": "A pest infestation that causes significant leaf damage in hot, dry conditions.",
        "cause": "Pest: Tetranychus urticae",
        "symptoms": "Tiny yellow or white speckles on leaves, fine webbing on undersides, bronzing of leaves.",
        "treatment": "Apply miticides or insecticidal soap, increase humidity, introduce predatory mites.",
        "severity": "Moderate"
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "status": "DISEASED",
        "description": "A fungal disease causing distinctive target-like spots on tomato leaves.",
        "cause": "Fungus: Corynespora cassiicola",
        "symptoms": "Brown spots with concentric rings and yellow halos giving a target appearance.",
        "treatment": "Apply fungicides (azoxystrobin or chlorothalonil), improve air circulation.",
        "severity": "Moderate"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "status": "DISEASED",
        "description": "A devastating viral disease spread by whiteflies that stunts plant growth.",
        "cause": "Virus: Tomato yellow leaf curl virus (TYLCV) transmitted by whiteflies",
        "symptoms": "Upward curling and yellowing of leaves, stunted growth, reduced fruit production.",
        "treatment": "No cure. Control whitefly populations, use resistant varieties, remove infected plants.",
        "severity": "High"
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "status": "DISEASED",
        "description": "A highly contagious viral disease that affects tomato plants worldwide.",
        "cause": "Virus: Tomato mosaic virus (ToMV)",
        "symptoms": "Mosaic pattern of light and dark green on leaves, leaf distortion, stunted growth.",
        "treatment": "No cure. Remove infected plants, disinfect tools, use resistant varieties.",
        "severity": "High"
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The tomato plant appears healthy with no signs of disease.",
        "cause": "N/A",
        "symptoms": "No symptoms detected.",
        "treatment": "Continue regular care and monitoring.",
        "severity": "None"
    },
}

SEVERITY_COLORS = {
    "None":     "✅",
    "Moderate": "⚠️",
    "High":     "🔴",
    "Critical": "🚨",
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, DENSE_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DENSE_UNITS, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


def get_disease_info(class_name):
    """Look up disease info, fallback to parsing the class name."""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    # Fallback: parse the class name
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in class_name.lower()
    return {
        "plant": plant,
        "disease": "None" if is_healthy else disease,
        "status": "HEALTHY" if is_healthy else "DISEASED",
        "description": f"{'Healthy ' + plant + ' plant.' if is_healthy else disease + ' detected on ' + plant + '.'}",
        "cause": "N/A" if is_healthy else "Unknown",
        "symptoms": "No symptoms." if is_healthy else "See agricultural guidelines.",
        "treatment": "Continue regular care." if is_healthy else "Consult local agricultural extension.",
        "severity": "None" if is_healthy else "Unknown"
    }


def predict(image_path, model_path, class_names_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(class_names_path) as f:
        class_names = json.load(f)

    model = CropDiseaseModel(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top1_idx = probs.argmax().item()
        confidence = probs[top1_idx].item()

    class_name = class_names[top1_idx]
    info = get_disease_info(class_name)
    severity_icon = SEVERITY_COLORS.get(info["severity"], "⚠️")

    print("\n" + "=" * 60)
    print("       CROP DISEASE DETECTION RESULT")
    print("=" * 60)
    print(f"  Image     : {image_path}")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print("-" * 60)
    print(f"  Plant     : {info['plant']}")
    print(f"  Status    : {severity_icon}  {info['status']}")
    print(f"  Disease   : {info['disease']}")
    print(f"  Severity  : {info['severity']}")
    print("-" * 60)
    print(f"  About     : {info['description']}")
    print(f"  Cause     : {info['cause']}")
    print(f"  Symptoms  : {info['symptoms']}")
    print(f"  Treatment : {info['treatment']}")
    print("=" * 60)

    return class_name, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True,  help="Path to leaf image")
    parser.add_argument("--model",       default="outputs/best_model.pth")
    parser.add_argument("--class_names", default="outputs/class_names.json")
    args = parser.parse_args()

    predict(args.image, args.model, args.class_names)
