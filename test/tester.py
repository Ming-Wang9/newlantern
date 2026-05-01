import json
import urllib.request
import time

# Load the public eval file
with open("test/relevant_priors_public.json") as f:
    data = json.load(f)

cases = data["cases"]
print(f"Total cases: {len(cases)}")

total_priors = sum(len(case["prior_studies"]) for case in cases)
print(f"Total priors to evaluate: {total_priors}")

# Build ground truth from truth list
gt = {}
for item in data["truth"]:
    gt[(item["case_id"], item["study_id"])] = item["is_relevant_to_current"]

print(f"Ground truth loaded: {len(gt)} labels")

# Strip labels and send to our API
clean_cases = []
for case in cases:
    clean_cases.append({
        "case_id": case["case_id"],
        "current_study": case["current_study"],
        "prior_studies": [
            {"study_id": p["study_id"], "study_description": p["study_description"]}
            for p in case["prior_studies"]
        ]
    })

payload = json.dumps({"cases": clean_cases}).encode()
req = urllib.request.Request(
    "http://localhost:8080/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("Sending request...")
t0 = time.time()
with urllib.request.urlopen(req, timeout=360) as resp:
    result = json.loads(resp.read())
print(f"Done in {time.time()-t0:.1f}s")

# Score
correct = 0
incorrect = 0
for pred in result["predictions"]:
    key = (pred["case_id"], pred["study_id"])
    if gt.get(key) == pred["predicted_is_relevant"]:
        correct += 1
    else:
        incorrect += 1

total = correct + incorrect
print(f"\nAccuracy: {correct}/{total} = {correct/total:.4f}")