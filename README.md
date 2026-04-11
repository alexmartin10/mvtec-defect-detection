# PatchCore — Industrial Defect Detection

Anomaly detection on industrial parts using the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark dataset. A PatchCore-inspired model is served via a REST API, containerized with Docker, and deployed on Google Cloud Run.

**Live API** → https://patchcore-api-122329804522.europe-west1.run.app/docs

---

## Results

Evaluated on the `bottle` category of MVTec AD (seed 42):

| Category | AUROC |
|---|---|
| broken_large | 1.000 |
| broken_small | 1.000 |
| contamination | 0.976 |
| **Overall** | **0.992** |

---

## Approach

PatchCore is a memory-based anomaly detection method that requires no training — only normal (defect-free) images.

**How it works:**

1. **Feature extraction** — a pretrained ResNet18 backbone extracts patch-level features from normal images (layers 2 & 3)
2. **Memory bank** — features from all normal images are stored in a memory bank, then subsampled (10%) for efficiency
3. **Anomaly scoring** — at inference, each patch of the test image is compared to its nearest neighbor in the memory bank via Euclidean distance. The maximum distance across all patches is the anomaly score
4. **Thresholding** — the decision threshold is set at the 95th percentile of anomaly scores on the training set

---

## Project Structure

```
mvtec-defect-detection/
├── api/
│   ├── main.py         # FastAPI endpoints
│   └── model.py        # PatchCore inference class
├── model/
│   └── patchcore.pt    # memory bank + threshold
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Request

Send an image file as `multipart/form-data`.

```bash
curl -X POST https://patchcore-api-122329804522.europe-west1.run.app/predict \
  -F "file=@your_image.png"
```

### Response

```json
{
  "score": 4.231,
  "anomaly": true,
  "threshold": 3.214
}
```

- `score` — anomaly score (higher = more anomalous)
- `anomaly` — `true` if defect detected, `false` otherwise
- `threshold` — decision threshold (95th percentile on training set)

---

## Run Locally

**With Docker (recommended):**

```bash
git clone https://github.com/your-username/mvtec-defect-detection
cd mvtec-defect-detection
docker build -t patchcore-api .
docker run -p 8000:8000 patchcore-api
```

**Without Docker:**

```bash
git clone https://github.com/alexmartin10/mvtec-defect-detection
cd mvtec-defect-detection
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Then open http://localhost:8000/docs

---

## Tech Stack

| | |
|---|---|
| Model | PatchCore (ResNet18 backbone, pretrained on ImageNet) |
| Framework | PyTorch + torchvision |
| API | FastAPI |
| Containerization | Docker |
| Deployment | Google Cloud Run |
| Dataset | MVTec AD — bottle category |

---

## Limitations

- Trained and evaluated on a single MVTec category (`bottle`) — generalization to other categories would require retraining the memory bank
- Random subsampling (10%) is used instead of the greedy coreset algorithm from the original PatchCore paper, trading a small amount of accuracy for speed
- No GPU at inference — suitable for low-throughput use cases
