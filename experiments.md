## Approach
Rule-based classifier using keyword matching on study descriptions.
No LLM used — pure Python pattern matching.

## What I built
- Extract modality (MRI, CT, XRAY, US, NM, MAMMO) from description
- Extract body region (BRAIN, CHEST, SPINE, BREAST etc.) from description
- If regions don't overlap → False
- If regions overlap and modalities compatible → True
- If can't determine → False (important!)

## Results
- Baseline (default True): 59.8%
- Added abbreviations: 75.1%
- Changed default to False: 89.2% ← final

## Key insight
Changing the default from True to False was the biggest single improvement,
jumping from 75% to 89% accuracy.

## Next improvements
- Add more medical abbreviations to keyword lists
- Handle PET/CT full body scans better
- Use a small ML classifier trained on the labeled data