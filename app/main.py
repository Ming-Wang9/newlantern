import re
import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Radiology Relevant Priors")

@app.get("/health")
def health():
    return {"status": "ok", "message": "Radiology Relevant Priors API is running"}

MODALITIES = {
    "MRI": ["MRI", "MR ", "MAGNETIC RESONANCE"],
    "CT": ["CT ", "CTA ", "COMPUTED TOMOGRAPHY", "CT/"],
    "XRAY": ["X-RAY", "XRAY", "RADIOGRAPH", "CHEST PA", "CHEST AP", "VIEWS", "VIEW"],
    "US": ["ULTRASOUND", "US ", "SONOGRAM", "ECHO", "DUPLEX"],
    "NM": ["NUCLEAR", "PET", "SPECT", "SPEC", "BONE SCAN", "NM ", "MUGA", "MYO", "PET/CT"],
    "MAMMO": ["MAMMOGRAM", "MAMMOGRAPHY", "MAM", "DIGITAL SCREENER", "TOMO"],
    "FLUORO": ["FLUOROSCOPY", "BARIUM", "SWALLOW"],
    "DEXA": ["DEXA", "BONE DENSITY", "DXA"],
}

def extract_modality(desc: str) -> str | None:
    d = desc.upper()
    for mod, patterns in MODALITIES.items():
        for p in patterns:
            if p in d:
                return mod
    return None

BODY_REGIONS = {
    "BRAIN": ["BRAIN", "HEAD", "CRANIAL", "SKULL", "ORBIT", "NEURO", "IAC", "PITUITARY", "SELLA"],
    "SPINE": ["SPINE", "SPINAL", "CERVICAL", "THORACIC", "LUMBAR", "VERTEBR", "CORD"],
    "NECK": ["NECK", "THYROID", "CAROTID", "LARYNX"],
    "CHEST": ["CHEST", "LUNG", "PULMONARY", "CARDIAC", "HEART", "RIB", "CORONARY", "MYO", "PERF", "AORTA", "MEDIASTIN"],
    "ABDOMEN": ["ABDOMEN", "ABDOMINAL", "LIVER", "PANCREAS", "KIDNEY", "RENAL", "GALLBLADDER", "SPLEEN", "BOWEL"],
    "PELVIS": ["PELVIS", "PELVIC", "BLADDER", "PROSTATE", "UTERUS", "HIP", "OVARY", "GYN"],
    "EXTREMITY": ["ARM", "LEG", "HAND", "FOOT", "KNEE", "SHOULDER", "WRIST", "ANKLE", "FEMUR", "TIBIA", "FINGER", "TOE", "ELBOW"],
    "BREAST": ["BREAST", "MAMMARY", "MAM", "DIGITAL SCREENER", "BILAT SCREEN"],
    "FACE": ["FACIAL", "MAXFACIAL", "SINUS", "SINUSES", "MANDIBLE", "TMJ"],
    "BONE": ["BONE DENSITY", "DEXA", "DXA"],
}

def extract_region(desc: str) -> set:
    d = desc.upper()
    found = set()
    for region, patterns in BODY_REGIONS.items():
        for p in patterns:
            if p in d:
                found.add(region)
    return found


MODALITY_COMPAT = {
    "MRI":   {"MRI", "CT"},
    "CT":    {"CT", "MRI", "XRAY"},
    "XRAY":  {"XRAY", "CT", "MRI"},
    "US":    {"US", "CT", "MRI"},
    "NM":    {"NM", "CT", "MRI"},
    "MAMMO": {"MAMMO", "US"},
    "FLUORO":{"FLUORO", "XRAY"},
}

def get_side(desc: str) -> str | None:
    d = desc.upper()
    if " RT " in d or " RIGHT " in d or "RT " in d:
        return "RIGHT"
    if " LT " in d or " LEFT " in d or "LT " in d:
        return "LEFT"
    if "BILAT" in d or "BILATERAL" in d:
        return "BILATERAL"
    return None

def is_relevant(current_desc: str, prior_desc: str) -> bool:
    cur_mod = extract_modality(current_desc)
    pri_mod = extract_modality(prior_desc)
    cur_reg = extract_region(current_desc)
    pri_reg = extract_region(prior_desc)

    # regions don't overlap → not relevant
    if cur_reg and pri_reg and not cur_reg & pri_reg:
        return False

    # regions overlap → check modality compatibility
    if cur_reg and pri_reg and cur_reg & pri_reg:
        if "BREAST" in cur_reg and "BREAST" in pri_reg:
            cur_side = get_side(current_desc)
            pri_side = get_side(prior_desc)
            if cur_side and pri_side:
                if cur_side != pri_side and "BILATERAL" not in (cur_side, pri_side):
                    return False
        if cur_mod and pri_mod:
            return pri_mod in MODALITY_COMPAT.get(cur_mod, {cur_mod})
        return True

    # can't determine → default False now
    return False


from pydantic import BaseModel

class StudyInfo(BaseModel):
    study_id: str
    study_description: str
    study_date: str | None = None

class CaseInput(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: StudyInfo
    prior_studies: list[StudyInfo]
 
class PredictRequest(BaseModel):
    challenge_id: str | None = None
    schema_version: int | None = None
    cases: list[CaseInput]


@app.post("/predict")
def predict(req: PredictRequest):
    predictions = []
    for case in req.cases:
        cur_desc = case.current_study.study_description
        for prior in case.prior_studies:
            relevant = is_relevant(cur_desc, prior.study_description)
            
            predictions.append({
                "case_id": case.case_id,
                "study_id": prior.study_id,
                "predicted_is_relevant": relevant
            })

    log.info(f"Processed {len(req.cases)} cases, {len(predictions)} predictions")
    return {"predictions": predictions}

        
