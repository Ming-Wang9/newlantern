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
    "CT": ["CT ", "CTA ", "COMPUTED TOMOGRAPHY"],
    "XRAY": ["X-RAY", "XRAY", "RADIOGRAPH", "CHEST PA", "CHEST AP"],
    "US": ["ULTRASOUND", "US ", "SONOGRAM"],
    "NM": ["NUCLEAR", "PET", "SPECT", "BONE SCAN"],
    "MAMMO": ["MAMMOGRAM", "MAMMOGRAPHY"],
    "FLUORO": ["FLUOROSCOPY", "BARIUM", "SWALLOW"],
}

def extract_modality(desc: str) -> str | None:
    d = desc.upper()
    for mod, patterns in MODALITIES.items():
        for p in patterns:
            if p in d:
                return mod
    return None

BODY_REGIONS = {
    "BRAIN": ["BRAIN", "HEAD", "CRANIAL", "SKULL", "ORBIT"],
    "SPINE": ["SPINE", "SPINAL", "CERVICAL", "THORACIC", "LUMBAR", "VERTEBR"],
    "NECK": ["NECK", "THYROID", "CAROTID"],
    "CHEST": ["CHEST", "LUNG", "PULMONARY", "CARDIAC", "HEART", "RIB"],
    "ABDOMEN": ["ABDOMEN", "ABDOMINAL", "LIVER", "PANCREAS", "KIDNEY", "RENAL"],
    "PELVIS": ["PELVIS", "PELVIC", "BLADDER", "PROSTATE", "UTERUS", "HIP"],
    "EXTREMITY": ["ARM", "LEG", "HAND", "FOOT", "KNEE", "SHOULDER", "WRIST", "ANKLE"],
    "BREAST": ["BREAST", "MAMMARY"],
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
        if cur_mod and pri_mod:
            return pri_mod in MODALITY_COMPAT.get(cur_mod, {cur_mod})
        return True  # regions match, modality unknown → assume relevant

    # can't determine
    return True


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

        
