# Sample CT Scan Images

Test images for manual and automated testing of the TumorVision API.

| File | Type | Source | Expected Class |
|------|------|--------|----------------|
| `ct_normal.png` | Synthetic grayscale | Generated | Normal |
| `ct_cyst.png` | Synthetic grayscale | Generated | Cyst |
| `ct_tumor.png` | Synthetic grayscale | Generated | Tumor |
| `ct_stone.png` | Synthetic grayscale | Generated | Stone |
| `normal_ct.jpg` | Real CT scan | Public GitHub repo | — |

## Usage

**Via React UI (http://localhost:8112):**
Drag and drop any image onto the upload zone and click Analyze.

**Via curl:**
```bash
curl -X POST http://localhost:8110/api/v1/predict \
  -F "file=@tests/sample_images/ct_tumor.png" | python3 -m json.tool
```

**Via Python:**
```python
import requests

with open("tests/sample_images/normal_ct.jpg", "rb") as f:
    r = requests.post("http://localhost:8110/api/v1/predict", files={"file": f})
    print(r.json()["predicted_class"], r.json()["confidence"])
```

> **Note:** Synthetic images were generated programmatically to simulate CT scan appearance.
> For production-quality testing, use the full dataset from
> [Kaggle: CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone).
