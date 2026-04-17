"""
AgriLens — Crop Disease Detection Web Server
Elegant Edition with Detailed Explanations
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory
import io

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "outputs", "best_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "outputs", "class_names.json")
STATIC_DIR      = os.path.join(BASE_DIR, "static")

IMG_SIZE      = 224
DENSE_UNITS   = 512
DROPOUT       = 0.5
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
#  Disease information database (Highly Detailed)
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "status": "DISEASED",
        "description": "Apple scab is a pervasive fungal disease that significantly impacts the visual quality and yield of apple trees. It primarily manifests as superficial lesions, making the fruit unmarketable and weakening the tree by causing premature leaf drop. The disease thrives in cool, wet spring weather and can quickly spread throughout an entire orchard if not managed properly.",
        "cause": "Caused by the ascomycete fungus Venturia inaequalis. The pathogen overwinters in fallen, diseased leaves. In the spring, spores are released and carried by wind and rain to newly developing leaves and fruit.",
        "symptoms": "Initial symptoms appear as pale, water-soaked, or chlorotic spots on leaves. These develop into distinct olive-green to dark brown or black velvety, scabby lesions. Infected fruit develop similar dark, corky scabs, often becoming deformed or cracking as they grow.",
        "treatment": "Immediate chemical intervention using fungicides such as Captan, Myclobutanil, or Mancozeb is often necessary during the primary infection period. Organic options include liquid copper soap or sulfur-based sprays applied at early bud break.",
        "prevention": "Implement strict orchard sanitation by racking and destroying fallen leaves in autumn or applying urea to accelerate leaf decomposition. Ensure proper pruning to open the canopy for better air circulation and sunlight penetration, which promotes faster drying of foliage. Plant scab-resistant apple cultivars like Enterprise or Liberty.",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "status": "DISEASED",
        "description": "Black Rot is a severe and structural disease affecting apple trees, capable of attacking the fruit, leaves, and bark. This disease not only destroys the immediate fruit harvest by causing a pervasive, firm rot, but it also creates sunken branch cankers that can slowly girdle and kill entire limbs or the tree itself.",
        "cause": "Triggered by the fungus Botryosphaeria obtusa. It aggressively colonizes dead tissue, mummified fruits left on the tree, and bark wounds caused by winter injury, insects, or improper pruning.",
        "symptoms": "On leaves, it begins as small purple spots that enlarge into characteristic 'frog-eye' lesions (brown spots with dark margins and lighter centers). On fruit, it causes a firm, dark brown rot that eventually turns the apple into a shriveled, black 'mummy'. On branches, it forms reddish-brown, sunken cankers.",
        "treatment": "Prune out and destroy infected branches and dead wood during the dormant season, cutting at least 15 inches below the canker. For fruit and leaf infections, apply appropriate fungicides (like captan or thiophanate-methyl) starting from the period right after petal fall.",
        "prevention": "Remove all mummified fruit from the tree and the ground. Avoid pruning during wet weather. Protect the bark from sunscald and winter injury, and promptly remove any fire blight or dead branches to eliminate entry points for the fungus.",
        "severity": "High"
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "status": "DISEASED",
        "description": "Cedar Apple Rust is a fascinating yet damaging heteroecious fungal disease, meaning it requires two entirely different host plants (apple trees and Eastern red cedar/junipers) to complete its complex two-year life cycle. Without both hosts in proximity, the disease cannot proliferate.",
        "cause": "Caused by the fungus Gymnosporangium juniperi-virginianae. Spores travel from gelatinous, horn-like galls on cedar trees in early spring directly to expanding apple leaves via wind currents.",
        "symptoms": "Appears as vivid, bright yellow to orange spots on the upper surfaces of apple leaves in late spring. By mid-summer, the undersides of these spots swell and produce small, hair-like tubes that release spores back to cedar trees. Severe infections cause premature defoliation and stunted, deformed fruit.",
        "treatment": "Protective fungicide applications (using sterol inhibitors like Myclobutanil) must be timed perfectly, beginning at the pink bud stage and continuing through petal fall until the cedar galls dry up.",
        "prevention": "The most effective cultural control is removing the alternate host (Eastern red cedars and susceptible junipers) within a 1-to-2 mile radius of the apple orchard, though this is rarely practical. Opt for planting highly rust-resistant apple varieties such as Cortland, Honeycrisp, or McIntosh.",
        "severity": "Moderate"
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "None",
        "status": "HEALTHY",
        "description": "This apple tree leaf exhibits excellent health and vitality. A vibrant, uniformly green canopy is the foundation of high fruit yield, effective photosynthesis, and natural defense against environmental stressors.",
        "cause": "Optimal synergy of proper nutrients, adequate soil moisture, balanced sunlight, and absence of active pathogenic infections or severe pest infestations.",
        "symptoms": "The leaf surface is smooth, featuring a rich, uniform green coloration without any telltale spots, lesions, chlorosis (yellowing), or structural deformations. Edges are intact and veins are clear.",
        "treatment": "No acute therapeutic treatment is required for this specimen.",
        "prevention": "Maintain this baseline health by adhering to a consistent regimen: conduct annual dormant pruning to optimize airflow, apply balanced NPK fertilizers based on soil tests, ensure deep but infrequent watering during dry spells, and practice regular scouting for early signs of pests or seasonal diseases.",
        "severity": "None"
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "status": "DISEASED",
        "description": "Leaf Scorch is a prominent foliar fungal disease affecting strawberries. When severe, it rapidly kills extensive tracts of leaf tissue, resembling serious heat or chemical scorch damage. Unchecked, it starves the plant of necessary carbohydrates and significantly diminishes both immediate berry size and subsequent crop yield.",
        "cause": "Caused by the fungus Diplocarpon earlianum. The infection requires prolonged periods of leaf wetness and typically occurs in dense, overgrown beds during periods of heavy dew or frequent rain.",
        "symptoms": "It manifests initially as small, irregular purple to dark red spots. These drastically expand and develop grayish-brown centers. As the spots fuse together, the entire leaf margin typically turns brown and curls upward, presenting a genuinely 'scorched' burnt appearance.",
        "treatment": "Applications of protective or systemic fungicides (e.g., Captan, Myclobutanil) offer control but must be timed appropriately. The best approach involves rotating chemistries to prevent resistance buildup.",
        "prevention": "Ensure planting beds have excellent drainage and adequate spacing. strictly avoid overhead sprinkler irrigation that heavily wets the leaves. Implement a vigorous renovation of the strawberry beds immediately after harvest—mowing the old leaves off to eliminate the fungal inoculum.",
        "severity": "Moderate"
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The blueberry plant demonstrates prime vigor. Healthy foliage indicates optimal root function, proper soil acidity, and active nutrient uptake, setting the stage for robust flowering and a bountiful berry harvest.",
        "cause": "Optimal acidic soil conditions (pH 4.5 to 5.5), excellent drainage, sufficient organic matter, and absence of major fungal pathogens like Mummy Berry or Botrytis.",
        "symptoms": "Leaves exhibit a deep, rich green color, are fully expanded without curling or marginal necrosis, and feature a slightly glossy texture on the upper surface.",
        "treatment": "No corrective treatments are necessary at this moment.",
        "prevention": "Ensure soil pH remains firmly within the acidic range; apply elemental sulfur if the pH creeps up. Apply a thick layer of pine bark mulch to retain moisture and keep roots cool. Water consistently, aiming for 1 to 2 inches per week, while being careful to avoid water-logged soil which invites Phytophthora root rot.",
        "severity": "None"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "status": "DISEASED",
        "description": "Powdery mildew is a ubiquitous fungal affliction that targets young, rapidly expanding leaves and green shoots. Unlike many fungal diseases that require free moisture, powdery mildew thrives in high humidity but relatively dry foliage conditions. It severely stunts new growth and diminishes overall tree vigor.",
        "cause": "Primarily caused by Podosphaera clandestina. The fungus overwinters in dormant buds or fallen leaves. Warm days followed by cool, humid nights promote rapid spore germination and spread.",
        "symptoms": "Characterized by white to pale gray, dusty, powdery fungal mats covering the surfaces of leaves and tender stems. Severely infected leaves may curl upward, pucker, become brittle, and drop prematurely. Fruit infections are rare in sour cherries but can cause a net-like russeting.",
        "treatment": "Apply eradicant or protectant fungicides strictly. Options include sulfur, potassium bicarbonate, or synthetic fungicides (myclobutanil or fenarimol). Ensure thorough coverage of the foliage for maximum efficacy.",
        "prevention": "Select planting sites with full, all-day sun exposure. Regularly prune trees to maintain an open canopy that allows direct sunlight and wind to penetrate completely, reducing ambient humidity around the leaves. Avoid excessive nitrogen fertilization, which forces highly susceptible flush growth.",
        "severity": "Moderate"
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The cherry foliage is functioning perfectly. A healthy canopy is critical for producing the carbohydrates necessary to size up the current year's fruit crop and form strong fruit buds for the subsequent season.",
        "cause": "Favorable environmental conditions, sufficient irrigation without waterlogging, and effective warding off of common pests like cherry fruit flies and fungal blights.",
        "symptoms": "Leaves are a bright, dark green, lacking any spotting, shot holes, or mildew residue. The foliage appears turgid, showing natural expansion without wrinkling.",
        "treatment": "No treatment required.",
        "prevention": "Maintain excellent orchard sanitation. Apply a dormant horticultural oil spray in late winter to smother overwintering pest eggs. Protect against birds as fruit ripens, and conduct routine visual inspections for early signs of cherry leaf spot or bacterial canker.",
        "severity": "None"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "status": "DISEASED",
        "description": "Gray Leaf Spot is one of the most economically devastating foliar diseases of corn worldwide. It rapidly destroys photosynthetic tissue during the critical grain-filling period, leading to massive reductions in ear size, kernel weight, and overall yield, as well as predisposing stalks to rot.",
        "cause": "Triggered by the fungus Cercospora zeae-maydis. It survives the winter on infested corn residue left on the soil surface. Warm, highly humid conditions and extensive periods of leaf wetness drive its rapid progression up the plant.",
        "symptoms": "Lesions begin as small, tan spots surrounded by yellow halos. As they mature, they expand into highly distinct, blocky, rectangular gray-to-tan lesions that run rigidly parallel to the leaf veins. In severe outbreaks, lesions coalesce, entirely blighting the leaf.",
        "treatment": "In high-risk situations (such as continuous corn cropping combined with wet weather), foliar application of a strobilurin or triazole fungicide is critical around the tasseling (VT) to silking (R1) stages to protect the ear leaf and upper canopy.",
        "prevention": "The foundation of management is crop rotation (at least one year away from corn) to break the disease cycle, and conventional tillage where appropriate to bury infected stubble. Above all, prioritize selecting and planting corn hybrids with high built-in genetic resistance to the disease.",
        "severity": "High"
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "status": "DISEASED",
        "description": "Common rust is a frequently observed fungal disease that dots the foliage of corn with brightly colored pustules. While its visual presentation is striking, it typically does not cause catastrophic yield losses unless the infection occurs extremely early in the plant's development and spreads massively.",
        "cause": "Caused by the fungus Puccinia sorghi. Interestingly, the spores cannot survive harsh winters; instead, the disease is carried northward each season on prevailing wind currents from tropical or subtropical regions.",
        "symptoms": "Presents as small, slightly elongated, raised oval pustules that erupt on both the upper and lower surfaces of the leaves. These pustules contain brick-red to golden-cinnamon colored powdery spores. As the leaf matures, the pustules turn dark brown to black as they enter a dormant overwintering stage.",
        "treatment": "For susceptible hybrids or sweet corn, fungicidal sprays may be warranted if pustules appear rapidly before the silking stage. Strobilurins and DMI fungicides are highly effective.",
        "prevention": "Planting commercially available corn hybrids that carry resistance genes is the primary and most robust defense. Timely planting early in the season often allows the corn to reach maturity before the airborne rust spores arrive in dense enough concentrations to cause harm.",
        "severity": "Moderate"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "status": "DISEASED",
        "description": "Northern Leaf Blight (NLB) is an aggressive, destructive fungal disease recognizable by its massive, eye-shaped lesions. It can rapidly consume leaf tissue, particularly targeting the crucial upper canopy that is responsible for feeding the developing ear. Severe infections before the silking stage can result in yield losses approaching 50%.",
        "cause": "Caused by the fungus Exserohilum turcicum. The pathogen rests in crop debris over winter. Spores are splashed by rain onto lower leaves and systematically progress upward during periods of moderate temperatures (65-80°F) and heavy dew.",
        "symptoms": "Begins as pale, long, elliptical gray-green spots. These expand into massive, cigar-shaped lesions measuring 1 to 6 inches in length, bordered by a distinctly darker margin. Under humid conditions, a dark fuzz of newly forming spores becomes visible within the lesions.",
        "treatment": "Fungicide applications are highly effective if applied aggressively when lesions are first spotted on leaves immediately below the ear leaf around the tasseling stage.",
        "prevention": "Strictly implement a one to two-year crop rotation (e.g., to soybeans) to starve the fungus. Plow under corn residue in fields with a history of NLB. Exclusively use corn hybrids with categorized resistance to Exserohilum turcicum races.",
        "severity": "High"
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)",
        "disease": "None",
        "status": "HEALTHY",
        "description": "This corn specimen is in peak robust health. The large, undamaged blade area ensures maximum capture of solar energy, driving robust stalk growth and enabling massive carbohydrate storage in the eventual grain harvest.",
        "cause": "Excellent soil fertility profiles (especially high nitrogen), optimal solar exposure, perfectly timed rainfall, and a high degree of genetic immunity.",
        "symptoms": "Leaves are broad, smooth, and feature a deep, brilliantly saturated green color. There is a complete absence of streaks, spotting, blistering, or edge burning.",
        "treatment": "No interventions are necessary.",
        "prevention": "Continue executing a rigorous crop management plan: implement routine soil testing to ensure necessary sidedress nitrogen applications, manage weed competition early, use pre-emergence herbicides, and scout routinely for pests like the European corn borer or fall armyworm.",
        "severity": "None"
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "status": "DISEASED",
        "description": "Black Rot is an incredibly aggressive, highly destructive fungal plague that poses an existential threat to grape harvests, particularly in warm, humid viticulture zones. If left unchecked, it can easily destroy an entire crop in a matter of days, attacking everything from green shoots to the grapes themselves.",
        "cause": "Originates from the fungus Guignardia bidwellii, which survives winter inside mummified, shriveled grapes clinging to vines or resting on the soil surface.",
        "symptoms": "Leaves display small, tan, circular spots bordered by prominent dark brown rings. As the disease jumps to the fruit, grapes develop whitish spots that rapidly expand, turning the entire berry soft and brown before it shrivels into a hard, coal-black, wrinkled mummy dotted with tiny black spore-producing pimples.",
        "treatment": "Immediate, rigorous application of systemic fungicides (such as myclobutanil or tebuconazole) combined with a protectant like mancozeb. Treatment schedules must be strictly adhered to, spraying every 10–14 days starting early in the season.",
        "prevention": "Meticulous canopy management is required: aggressively thin leaves around the fruit cluster to maximize air circulation. Practice ruthless sanitation by physically pruning out and destroying every single mummified grape from the vines and vineyard floor during winter dormancy.",
        "severity": "High"
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "status": "DISEASED",
        "description": "Esca is an intensely complex and poorly understood disease of the internal vine wood that manifests bizarre and sudden external symptoms. It slowly rots the interior vascular tissue of mature vines, restricting water flow, and can cause sudden, devastating vine collapse mid-season.",
        "cause": "Caused by a volatile complex of several fungi, predominantly Phaeomoniella chlamydospora and Phaeoacremonium aleophilum. Pathogens primarily enter the vine through pruning wounds over years.",
        "symptoms": "A wildly distinct, stark 'tiger-stripe' pattern emerges on leaves, as areas between the veins turn bright yellow or deep red while the main veins remain green. Berries may develop pinpoint dark purplish-black spots, giving a 'measles' appearance. Cross-sectioning the trunk reveals a highly compromised, dark, sponge-like internal wood decay.",
        "treatment": "Because the infection resides deep within the structural wood, there are virtually no chemical cures. The only recourse is radical surgical pruning—cutting the trunk down dramatically below the lowest sign of internal dark staining—or fully uprooting the vine.",
        "prevention": "Treat all large pruning wounds immediately with a protective fungicidal paste or mastic. Refrain entirely from making large cuts during wet, high-risk weather. Implement 'double pruning' techniques to manage the entry of fungi.",
        "severity": "High"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Grape Leaf Blight",
        "status": "DISEASED",
        "description": "Grape Leaf Blight is a foliar fungal infection that directly compromises the plant's photosynthetic engine. In severe, unmanaged cases, the resulting rapid defoliation exposes the grapes to devastating sunscald and completely stalls sugar accumulation leading into the harvest.",
        "cause": "Rooted in the fungus Pseudocercospora vitis. It lies dormant in fallen leaves on the vineyard floor over the winter, releasing infectious spores during spring rains.",
        "symptoms": "Manifests as irregular, dark purplish-brown lesions on leaves, often surrounded by diffuse yellow halos. As the season progresses, these spots expand, dry out, and merge. The leaf rapidly yellows, shrivels, and drops off prematurely.",
        "treatment": "Applications of copper-based bactericides or systemic fungicides typically offer excellent control if initiated when lesions first appear or just before the anticipated rainy season.",
        "prevention": "Ensure optimal vine spacing and aggressive canopy thinning to promise rapid drying. Diligently plow fallen leaves into the soil after the first frost, or remove them entirely to utterly disrupt the overwintering inoculum.",
        "severity": "Moderate"
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "None",
        "status": "HEALTHY",
        "description": "This grapevine leaf is an exemplar of structural virility. Healthy leaves are the powerhouses that synthesize the complex sugars and robust phenolic compounds necessary for premium grape quality.",
        "cause": "Optimal viticulture management including precise irrigation, balanced nutrition, excellent airflow management, and proactive preventative sulfur/copper sprays.",
        "symptoms": "Broad, vibrant green leaves with distinctly sharp lobes and lobes free of chlorosis, necrosis, curling, or pest webbing. The canopy looks visually uniform and lush.",
        "treatment": "No prescriptive intervention required.",
        "prevention": "Maintain a disciplined preventive sulfur spray program from early bud break to combat unseen powdery mildew. Ensure diligent shoot thinning and leaf removal in the fruiting zone to maintain microclimate perfection.",
        "severity": "None"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Citrus Greening",
        "status": "DISEASED",
        "description": "Huanglongbing (HLB), globally known as Citrus Greening, is an apocalyptic, incurable bacterial disease that has decimated the citrus industry worldwide. It ruthlessly chokes off the vascular system (phloem) of the tree, leading to slow but inevitable death.",
        "cause": "Triggered by the unculturable bacterium Candidatus Liberibacter asiaticus, which is efficiently transmitted tree-to-tree by a tiny flying insect vector known as the Asian citrus psyllid.",
        "symptoms": "Visuals include asymmetrical, blotchy yellow mottling of the leaves—unlike the symmetrical yellowing caused by nutrient deficiencies. Shoots may present bizarrely yellow ('yellow dragon'). Most destructively, fruit grows small, lopsided, remains partially green (hence the name), and possesses a starkly bitter, unusable taste.",
        "treatment": "Tragically, there is currently no known cure for Citrus Greening. Intensive, high-dose nutritional therapy (foliar feeding) and heat treatments can temporarily mask symptoms and slightly prolong fruit production, but tree death is certain.",
        "prevention": "Mandatory, relentless control of the Asian citrus psyllid using aggressive, coordinated insecticide programs. Aggressively scout and immediately uproot and incinerate any infected trees. Utilize entirely enclosed, screen-house nurseries to guarantee the production of completely disease-free planting stock.",
        "severity": "Critical"
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "status": "DISEASED",
        "description": "Bacterial spot is a tenacious, highly impactful disease specifically harmful to stone fruits. It ravages canopy foliage leading to severe early defoliation (weakening the tree for winter) and causes hideous pitting on the fruit that ruins fresh-market viability.",
        "cause": "Instigated by the highly motile bacterium Xanthomonas arboricola pv. pruni. It overwinters inside specialized pockets in the twigs and erupts via wind-driven rain and heavy spring dews.",
        "symptoms": "Leaves develop tiny, angular, water-soaked spots that quickly turn black. The centers of these spots dry up and drop out, resulting in a ragged 'shot-hole' appearance. The fruit develops deep, dark, sunken, cracking scabs, often weeping a gummy exudate.",
        "treatment": "Because it's a bacterium, standard fungal therapies are useless. Implement a rigid spray regimen incorporating fixed copper sprays or the antibiotic oxytetracycline starting from late dormancy through petal fall and early fruit development.",
        "prevention": "By far the most successful strategy is planting genetically resistant or highly tolerant peach varieties. Ensure windbreaks are maintained to reduce driving, sand-blasting winds that create micro-wounds on the leaves where bacteria violently enter.",
        "severity": "Moderate"
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "None",
        "status": "HEALTHY",
        "description": "The foliage of this peach tree is thriving. Exceptional leaf health is paramount in stone fruits for accumulating enormous sugar reserves and supporting robust bud development for the following spring.",
        "cause": "Ideal growing environment, characterized by excellent soil drainage, appropriate seasonal pruning, and a disciplined preventative maintenance program.",
        "symptoms": "Leaves are perfectly elongated, possessing a rich, smooth green hue with sharply defined margins, absent of the curling often associated with aphids or leaf curl, and exhibiting zero shot-holes.",
        "treatment": "No treatment required.",
        "prevention": "Adhere to the standard preventative dormant spray regimen involving copper to stave off unseen overwintering pathogens like Peach Leaf Curl. Maintain a clear, weed-free zone around the trunk and monitor rigorously for destructive borers.",
        "severity": "None"
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "status": "DISEASED",
        "description": "Early Blight is an extremely prevalent, almost inevitable fungal disease attacking tomatoes globally. While it usually begins late in the season despite its name, it causes severe defoliation starting from the bottom up, resulting in massive reductions in yield and exposing tender fruit to severe sunscald.",
        "cause": "Initiated by the fungus Alternaria solani. The pathogen lives deeply in the soil on decaying plant tissue and is relentlessly splashed onto the lowest, older leaves by heavy rain or aggressive overhead watering.",
        "symptoms": "Noticeable as dark, circular, leathery brown spots that contain distinct, diagnostic concentric rings resembling a miniature archery target. These concentric lesions are heavily surrounded by bright yellow halos. The disease aggressively marches upward, turning entire lower leaves totally yellow before they drop.",
        "treatment": "Apply a strict prophylactic and reactive fungicide program utilizing chlorothalonil, mancozeb, or organic copper solutions immediately at the very first sign of spotting, repeated every 7 to 10 days.",
        "prevention": "Implement deep organic mulching (like straw or wood chips) to create a physical barrier preventing soil splash. Strictly utilize drip irrigation to keep upper foliage bone-dry. Immediately prune off and incinerate the lowest branches (up to the first fruit cluster) to improve airflow.",
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "status": "DISEASED",
        "description": "Late blight is famously known as the devastating organism responsible for the historic Irish Potato Famine. It is a highly contagious, unbelievably fast-moving plant pandemic that can systematically reduce a lush, green tomato field to a blackened, rotting wasteland in mere days.",
        "cause": "Triggered by the fearsome oomycete (water mold) Phytophthora infestans. Spores travel for miles on storm winds. It requires cool, extensively moist, and incredibly humid conditions to ignite widespread infection.",
        "symptoms": "Starts as irregularly shaped, pale, water-soaked, 'greasy' looking lesions on leaves. Under high humidity, a stark white, fuzzy mold blooms violently on the leaf undersides. Stem lesions turn black. Fruit develops massive, rock-hard, dark greasy blotches that eventually rot the entire tomato.",
        "treatment": "Infection spreads so rapidly that there is almost no cure once established. Preventative, powerful systemic fungicides (like metalaxyl or modern specialized treatments) must be running on high-risk crops constantly during favorable weather.",
        "prevention": "Never leave potato tubers or tomato remnants in the ground over winter. Constantly monitor regional agricultural alert systems. Uproot, bag, and permanently destroy (do not compost) any infected plant immediately upon identification to prevent community spread.",
        "severity": "Critical"
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "status": "DISEASED",
        "description": "Tomato Mosaic Virus (ToMV) is an aggressively contagious viral pathogen that hijacks the cellular machinery of the plant. It severely stunts physiological development, reducing yields to zero and churning out bizarrely deformed, highly unappetizing fruit.",
        "cause": "A highly stable, resilient virus transmitted entirely mechanically. It spreads via contaminated hands, pruning shears, stakes, clothing, and even seemingly clean seeds. It does not require an insect vector.",
        "symptoms": "Foliage exhibits a vivid, swirling 'mosaic' pattern of intertwined light, dark green, and yellow patches. Leaves forcefully curl, become heavily wrinkled, and occasionally narrow into bizarre, stringy shapes (fern-leafing). Growth stalls radically.",
        "treatment": "Because it is a virus deeply integrated into the plant cells, chemical treatments are mathematically impossible. There is absolute zero cure.",
        "prevention": "Absolute sanitation is mandatory. Wash hands rigorously with soap or milk before handling plants. Sanitize all cutting tools constantly in bleach solutions. Absolutely ban any tobacco use near the crop, as related viruses linger in commercial tobacco. Solely purchase certified virus-free seeds or strongly resistant varieties.",
        "severity": "High"
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "None",
        "status": "HEALTHY",
        "description": "This tomato plant is in a state of absolute excellence, exhibiting optimal photosynthetic capacity. A vibrant canopy is essential for feeding the massive energy demands of generating heavy, robust, sugar-filled fruits.",
        "cause": "Perfect synergy of well-drained, deeply fertile soil, flawless irrigation management that maintains consistent moisture without soaking leaves, and a stellar preventative maintenance regime.",
        "symptoms": "Leaves possess a deep, lush, solid green color. They display normal, broad expansion without unnatural cupping, rolling, or microscopic spotting. The main stem is thick, sturdy, and free of discoloration.",
        "treatment": "No prescriptive treatments are currently necessary.",
        "prevention": "Ensure continuing success by maintaining steady drip-irrigation to prevent fruit blossom-end rot. Consistently prune out non-productive 'suckers' to boost canopy airflow and actively apply calcium-rich amendments.",
        "severity": "None"
    }
}

# ─────────────────────────────────────────────
#  Fallback logic for unlisted diseases
# ─────────────────────────────────────────────
def get_disease_info(class_name):
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in class_name.lower()
    
    return {
        "plant": plant,
        "disease": "None" if is_healthy else disease,
        "status": "HEALTHY" if is_healthy else "DISEASED",
        "description": f"The model has detected {'a healthy ' + plant + ' plant' if is_healthy else 'signs of ' + disease + ' on ' + plant + ' leaves'}. This requires careful observation.",
        "cause": "Environmental stress, fungal spores, or bacterial infiltration typical for this plant species." if not is_healthy else "Optimal growing conditions.",
        "symptoms": "Visible changes in leaf texture, color spotting, or general wilting." if not is_healthy else "Smooth, vibrant, unblemished foliage with proper turgidity.",
        "treatment": "Consult local agricultural extensions for specific chemical or organic control protocols." if not is_healthy else "No treatment required. Maintain current care routines.",
        "prevention": "Ensure good soil drainage, adequate plant spacing for airflow, and practice crop rotation." if not is_healthy else "Routine soil testing, balanced watering, and vigilant pest scouting.",
        "severity": "None" if is_healthy else "Moderate"
    }

# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────

# Standard inference transform: Resize short-side to 256, then center-crop 224
# This preserves aspect ratio and focuses on the leaf subject
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Test-Time Augmentation (TTA) transforms for robust predictions
tta_transforms = [
    # Original (center crop)
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
    # Slight rotation +15°
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation((15, 15)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
    # Slight rotation -15°
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomRotation((-15, -15)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
    # Slightly larger crop (zoom out)
    transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]),
]

CONFIDENCE_THRESHOLD = 0.40  # Minimum 40% confidence to make a prediction

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

# ─────────────────────────────────────────────
#  Initialization
# ─────────────────────────────────────────────
print("[INFO] Loading Elegant Model Backend...")
with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CropDiseaseModel(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"[INFO] Model loaded successfully on {device} — {len(class_names)} classes")

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        # EXIF Fix: Rotates image correctly if taken on a smartphone
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")

        # ── Test-Time Augmentation (TTA) ──────────────────
        # Average predictions from multiple augmented views
        # for more robust classification on real-world images
        all_probs = []
        with torch.no_grad():
            for tta_t in tta_transforms:
                tensor = tta_t(img).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                all_probs.append(probs)

            # Average all TTA predictions
            avg_probs = torch.stack(all_probs).mean(dim=0)
            top1_idx = avg_probs.argmax().item()
            confidence = avg_probs[top1_idx].item()

            top3_probs, top3_indices = torch.topk(avg_probs, min(3, len(avg_probs)))
            top3 = []
            for prob, idx in zip(top3_probs, top3_indices):
                cn = class_names[idx.item()]
                info = get_disease_info(cn)
                top3.append({
                    "class_name": cn,
                    "confidence": round(prob.item() * 100, 2),
                    "plant": info["plant"],
                    "disease": info["disease"],
                })

        # ── Confidence Threshold Check ────────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "success": True,
                "prediction": {
                    "class_name": "Unknown",
                    "confidence": round(confidence * 100, 2),
                    "plant": "Unrecognized",
                    "disease": "N/A",
                    "status": "UNCERTAIN",
                    "description": f"The model could not confidently identify this image (confidence: {confidence*100:.1f}%). This may not be a supported plant leaf, or the image quality may be insufficient. Please try uploading a clearer, close-up image of the leaf against a plain background.",
                    "cause": "Image may not match any of the 38 trained disease/plant classes, or the leaf is not clearly visible.",
                    "symptoms": "N/A",
                    "treatment": "Please try again with a clearer image. Ensure the leaf fills most of the frame and is well-lit.",
                    "prevention": "For best results, photograph individual leaves against a contrasting background with good lighting.",
                    "severity": "None",
                },
                "top3": top3,
            })

        class_name = class_names[top1_idx]
        info = get_disease_info(class_name)

        result = {
            "success": True,
            "prediction": {
                "class_name": class_name,
                "confidence": round(confidence * 100, 2),
                "plant": info["plant"],
                "disease": info["disease"],
                "status": info["status"],
                "description": info["description"],
                "cause": info["cause"],
                "symptoms": info["symptoms"],
                "treatment": info["treatment"],
                "prevention": info["prevention"],
                "severity": info["severity"],
            },
            "top3": top3,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Analysis sequence failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
