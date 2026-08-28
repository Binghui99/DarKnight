# Dark Web Captcha OCR Benchmark Dataset

A benchmark dataset for evaluating OCR and vision-language models on real-world CAPTCHA images collected from four regional dark web sites. Each subset reflects a distinct CAPTCHA style encountered in on real and operational dark web cites.

## Subsets

| Name | Samples | Type | Format | Resolution |
|---|---|---|---|---|
| `changAn` | 500 | Chinese characters · Alphanumeric · Math (mixed) | PNG | 240 × 80 |
| `freecity` | 500 | Alphanumeric | PNG | 200 × 50 |
| `mobile_store` | 500 | Alphanumeric | JPG | 60 × 30 |
| `russian_market` | 500 | Alphanumeric | PNG | 100 × 32 |

<!-- > **Note on `freecity`:** This subset contains only 22 unique images repeated across 500 samples (11 unique labels). It reflects a real-world CAPTCHA source with very limited variation. Accuracy on this subset will be inflated for any model that encounters repeated images, and should be interpreted with caution. It is included for completeness. -->

## Repository Structure

```
datasets_final/
├── changAn.json
├── changAn/          ← raw PNG images
├── freecity.json
├── freecity/         ← raw PNG images
├── mobile_store.json
├── mobile_store/     ← raw JPG images
├── russian_market.json
└── russian_market/   ← raw PNG images
```

## JSON Format

Each `.json` file is a list of entries with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | int / str | Unique sample identifier |
| `label` | str | Ground-truth CAPTCHA text |
| `image_path` | str | Original relative path at collection time |
| `b64_string` | str | Base64-encoded image (self-contained, no path needed) |

Example entry:

```json
{
  "id": 1,
  "label": "只别华才",
  "image_path": "changAn/captcha_Jae99g719B9xTBOKH8mm.png",
  "b64_string": "/9j/4AAQ..."
}
```

## Usage

The dataset is self-contained via `b64_string` — no image path resolution needed.

```python
import json
import base64
from io import BytesIO
from PIL import Image

with open("changAn.json", encoding="utf-8") as f:
    dataset = json.load(f)

entry = dataset[0]
print("Label:", entry["label"])

# Decode image from base64
b64 = entry["b64_string"]
if "," in b64:                        # strip data: URI header if present
    b64 = b64.split(",", 1)[1]
img = Image.open(BytesIO(base64.b64decode(b64)))
img.show()
```

To load raw image files instead:

```python
from pathlib import Path
from PIL import Image

img = Image.open(Path("changAn") / Path(entry["image_path"]).name)
```

## License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Free for non-commercial research and educational use with attribution.
Commercial use requires explicit permission from the authors.

## Citation

Citation will be added upon paper acceptance.

## Ethical Note

This dataset is intended for security research, robustness evaluation, and academic benchmarking of OCR systems. The images are collected from publicly accessible CAPTCHA endpoints. We do not condone using this dataset to build or improve production CAPTCHA-solving services.
