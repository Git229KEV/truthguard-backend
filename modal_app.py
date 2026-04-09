"""
TruthGuard API - Modal deployment
"""
import modal
from modal import Image, App, fastapi_endpoint, Secret
from fastapi import UploadFile, File
import os
import io
import base64
import re
import threading

app = App("truthguard-api", secrets=[Secret.from_name("HF_TOKEN")])

MODAL_IMAGE = Image.debian_slim(python_version="3.11").pip_install(
    "fastapi>=0.110.0",
    "uvicorn>=0.29.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "pillow>=10.0.0",
    "tavily-python>=0.3.0",
    "google-generativeai>=0.7.0",
    "python-multipart>=0.0.9",
    "python-dotenv>=1.0.0",
    "sentencepiece>=0.1.0",
    "protobuf>=3.20.1",
    "huggingface_hub>=0.21.0",
    "psutil>=5.9.0",
)

MODEL_DIR = "/model-cache"
SIGLIP_MODEL_PATH = f"{MODEL_DIR}/siglip"
XLM_MODEL_PATH = f"{MODEL_DIR}/xlm"

SIGLIP_HF_REPO = "Kevin229/final_siglip_model"
XLM_HF_REPO = "Kevin229/final_xlm_roberta_model"

genai_client = None
tavily_client = None
siglip_model = None
siglip_processor = None
xlm_model = None
xlm_tokenizer = None
torch = None
models_ready = {"status": "loading", "siglip": False, "xlm": False}
models_loaded = False
load_lock = threading.Lock()


def get_token():
    return os.environ.get("HF_TOKEN", "")


def download_models():
    global siglip_model, siglip_processor, xlm_model, xlm_tokenizer, torch, models_ready, models_loaded
    
    if models_loaded:
        return
    
    with load_lock:
        if models_loaded:
            return
            
        print("[INFO] Downloading models from HuggingFace...")
        
        os.makedirs(SIGLIP_MODEL_PATH, exist_ok=True)
        os.makedirs(XLM_MODEL_PATH, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=SIGLIP_HF_REPO,
                local_dir=SIGLIP_MODEL_PATH,
                local_dir_use_symlinks=False,
                token=get_token(),
            )
            print("[OK] SigLIP downloaded")
        except Exception as e:
            print(f"[FAIL] SigLIP download: {e}")
        
        try:
            snapshot_download(
                repo_id=XLM_HF_REPO,
                local_dir=XLM_MODEL_PATH,
                local_dir_use_symlinks=False,
                token=get_token(),
            )
            print("[OK] XLM-R downloaded")
        except Exception as e:
            print(f"[FAIL] XLM download: {e}")
        
        print("[INFO] Loading PyTorch...")
        try:
            import torch
            device = torch.device("cpu")
            torch.set_default_device(device)
        except Exception as e:
            print(f"[FAIL] PyTorch: {e}")
        
        if torch:
            try:
                from transformers import SiglipProcessor, SiglipForImageClassification
                siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_PATH)
                siglip_model = SiglipForImageClassification.from_pretrained(
                    SIGLIP_MODEL_PATH,
                    torch_dtype=torch.float32
                )
                siglip_model.eval()
                print("[OK] SigLIP loaded")
                models_ready["siglip"] = True
            except Exception as e:
                print(f"[FAIL] SigLIP load: {e}")
            
            try:
                from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
                xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_MODEL_PATH)
                xlm_model = XLMRobertaForSequenceClassification.from_pretrained(
                    XLM_MODEL_PATH,
                    torch_dtype=torch.float32
                )
                xlm_model.eval()
                print("[OK] XLM-R loaded")
                models_ready["xlm"] = True
            except Exception as e:
                print(f"[FAIL] XLM load: {e}")
        
        models_ready["status"] = "ready"
        models_loaded = True
        print("[INFO] Models ready")


@app.function(image=MODAL_IMAGE, timeout=600, memory=4096, gpu="any")
def init_models():
    download_models()
    return {"status": "ready", "siglip": models_ready["siglip"], "xlm": models_ready["xlm"]}


def parse_visual_label(raw_label):
    label_lower = str(raw_label).lower()
    if "label_0" in label_lower or label_lower == "0" or "fake" in label_lower:
        return "RUMOR"
    if "label_1" in label_lower or label_lower == "1" or "true" in label_lower or "real" in label_lower:
        return "NON-RUMOR"
    if any(x in label_lower for x in ["not_rumor", "non_rumor", "real", "true", "legit"]):
        return "NON-RUMOR"
    if any(x in label_lower for x in ["fake", "false", "rumor", "misinformation", "label_0"]):
        return "RUMOR"
    return "UNKNOWN"


@app.function(image=MODAL_IMAGE, timeout=300, memory=4096)
@fastapi_endpoint(method="GET")
def health():
    download_models()
    return {
        "status": "healthy",
        "models_status": models_ready["status"],
        "siglip_loaded": models_ready["siglip"],
        "xlm_loaded": models_ready["xlm"],
        "gemini_available": genai_client is not None,
        "tavily_available": tavily_client is not None,
    }


@app.function(image=MODAL_IMAGE, timeout=300, memory=4096)
@fastapi_endpoint(method="POST")
async def analyze(file: UploadFile = File(...)):
    from PIL import Image
    import torch
    
    download_models()
    
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "File must be an image"}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_base64 = base64.b64encode(contents).decode()
        
        results = {
            "visual": "UNAVAILABLE",
            "text": "UNAVAILABLE",
            "tavily": "UNAVAILABLE",
            "gemini": "UNAVAILABLE",
            "final": "NON-RUMOR",
            "sources": [],
            "confidence": 0.5,
        }
        
        if torch and siglip_model and siglip_processor:
            try:
                inputs = siglip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    logits = siglip_model(**inputs).logits
                pred = logits.argmax().item()
                raw_vis = siglip_model.config.id2label.get(pred, str(pred))
                results["visual"] = "RUMOR" if any(x in raw_vis.lower() for x in ["fake", "false", "rumor"]) else "NON-RUMOR"
                print(f"[SigLIP] {results['visual']}")
            except Exception as e:
                print(f"SigLIP Error: {e}")
        
        if xlm_model and xlm_tokenizer:
            try:
                text_input = "This is a news article about current events"
                inputs_xlm = xlm_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    logits = xlm_model(**inputs_xlm).logits
                text_pred = logits.argmax().item()
                raw_txt = xlm_model.config.id2label.get(text_pred, str(text_pred))
                results["text"] = "RUMOR" if "0" in raw_txt or "fake" in raw_txt.lower() else "NON-RUMOR"
                print(f"[XLM-R] {results['text']}")
            except Exception as e:
                print(f"XLM Error: {e}")
        
        verdicts = [v for v in [results["visual"], results["text"]] if v in ["RUMOR", "NON-RUMOR"]]
        r_count = verdicts.count("RUMOR")
        results["final"] = "RUMOR" if r_count > len(verdicts) / 2 else "NON-RUMOR"
        results["confidence"] = max(r_count, len(verdicts) - r_count) / len(verdicts) if verdicts else 0.5
        
        return results
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return {"error": str(e)}


@app.local_entrypoint()
def main():
    result = init_models.remote()
    print(f"Init result: {result}")
