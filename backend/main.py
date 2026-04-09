from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import json
import os
import time
import threading
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] pytesseract not available, using fallback OCR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path, override=True)

print(f"[INFO] Loading .env from: {env_path}")
print(f"[INFO] GOOGLE_API_KEY: {'set' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
print(f"[INFO] TAVILY_API_KEY: {'set' if os.getenv('TAVILY_API_KEY') else 'NOT SET'}")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

default_siglip_path = os.path.join(BASE_DIR, "models", "final_siglip_model")
default_xlm_path = os.path.join(BASE_DIR, "models", "final_xlm_roberta_model")

SIGLIP_PATH = os.getenv("SIGLIP_MODEL_PATH")
if SIGLIP_PATH:
    SIGLIP_PATH = SIGLIP_PATH.strip('"').strip("'")
if not SIGLIP_PATH:
    SIGLIP_PATH = default_siglip_path

XLM_PATH = os.getenv("XLM_ROBERTA_MODEL_PATH")
if XLM_PATH:
    XLM_PATH = XLM_PATH.strip('"').strip("'")
if not XLM_PATH:
    XLM_PATH = default_xlm_path

print(f"[INFO] SIGLIP_PATH: {SIGLIP_PATH}")
print(f"[INFO] XLM_PATH: {XLM_PATH}")

def list_gdrive_folder_contents(folder_id):
    try:
        import gdown
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        files = gdown.list_folder(url)
        return files
    except Exception as e:
        print(f"[WARN] Cannot list folder: {e}")
        return []

def download_from_huggingface(repo_id, dest_path):
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(dest_path, exist_ok=True)
        print(f"[INFO] Downloading from HuggingFace: {repo_id}")
        
        kwargs = {"local_dir": dest_path, "local_dir_use_symlinks": False}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        
        snapshot_download(repo_id=repo_id, **kwargs)
        
        print(f"[INFO] Contents after download:")
        for root, dirs, files in os.walk(dest_path):
            for f in files:
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path) / (1024*1024)
                    print(f"  {os.path.relpath(full_path, dest_path)} ({size:.1f} MB)")
                except:
                    pass
        
        model_files = []
        for root, dirs, files in os.walk(dest_path):
            for f in files:
                if f in ['model.safetensors', 'pytorch_model.bin', 'config.json', 'tokenizer.json']:
                    model_files.append(os.path.join(root, f))
        
        if model_files:
            print(f"[INFO] Found model files: {model_files}")
        
        return len(model_files) > 0
    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")
        return False

SIGLIP_HF_REPO = os.getenv("SIGLIP_HF_REPO", "Kevin229/final_siglip_model")
XLM_HF_REPO = os.getenv("XLM_HF_REPO", "Kevin229/final_xlm_roberta_model")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"[INFO] SIGLIP_HF_REPO: {SIGLIP_HF_REPO}")
print(f"[INFO] XLM_HF_REPO: {XLM_HF_REPO}")

def check_model_valid(model_path, min_size_mb=50):
    for root, dirs, files in os.walk(model_path):
        for f in files:
            if f in ['model.safetensors', 'pytorch_model.bin']:
                size_mb = os.path.getsize(os.path.join(root, f)) / (1024*1024)
                print(f"[INFO] Found {f} at {root}: {size_mb:.1f} MB")
                if size_mb < min_size_mb:
                    print(f"[WARN] {f} is too small ({size_mb:.1f} MB), likely corrupted")
                    return False
    return True

siglip_exists = os.path.exists(SIGLIP_PATH) and os.listdir(SIGLIP_PATH)
xlm_exists = os.path.exists(XLM_PATH) and os.listdir(XLM_PATH)

print(f"[INFO] SigLIP dir exists: {os.path.exists(SIGLIP_PATH)}, has files: {siglip_exists}")
print(f"[INFO] XLM dir exists: {os.path.exists(XLM_PATH)}, has files: {xlm_exists}")

siglip_valid = check_model_valid(SIGLIP_PATH) if siglip_exists else False
xlm_valid = check_model_valid(XLM_PATH) if xlm_exists else False

if not siglip_exists or not siglip_valid:
    print(f"[INFO] SigLIP not found or invalid, downloading from HuggingFace...")
    import shutil
    if os.path.exists(SIGLIP_PATH):
        shutil.rmtree(SIGLIP_PATH)
    download_from_huggingface(SIGLIP_HF_REPO, SIGLIP_PATH)
    siglip_exists = os.path.exists(SIGLIP_PATH) and os.listdir(SIGLIP_PATH)
    siglip_valid = check_model_valid(SIGLIP_PATH)
    print(f"[INFO] SigLIP download result - exists: {siglip_exists}, valid: {siglip_valid}")

if not xlm_exists or not xlm_valid:
    print(f"[INFO] XLM not found or invalid, downloading from HuggingFace...")
    import shutil
    if os.path.exists(XLM_PATH):
        shutil.rmtree(XLM_PATH)
    download_from_huggingface(XLM_HF_REPO, XLM_PATH)
    xlm_exists = os.path.exists(XLM_PATH) and os.listdir(XLM_PATH)
    xlm_valid = check_model_valid(XLM_PATH)
    print(f"[INFO] XLM download result - exists: {xlm_exists}, valid: {xlm_valid}")

genai_client = None
tavily_client = None
tavily_cache = {}

if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai_lib
        genai_lib.configure(api_key=GOOGLE_API_KEY)
        genai_client = genai_lib
        print("[OK] Gemini configured")
    except Exception as e:
        print(f"[FAIL] Gemini config error: {e}")

if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        print("[OK] Tavily configured")
    except Exception as e:
        print(f"[FAIL] Tavily config error: {e}")

device = "cpu"
torch = None
try:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] PyTorch loaded, device: {device}")
except Exception as e:
    print(f"[FAIL] PyTorch error: {e}")

siglip_model = None
siglip_processor = None
xlm_tokenizer = None
xlm_model = None

models_ready = {
    "status": "loading",
    "siglip": False,
    "xlm": False
}

def get_available_memory_mb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except:
        return 999999

def find_model_dir(base_path):
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            print(f"[DEBUG] Checking dir: {root}, files: {files}")
            if 'config.json' in files and any(f in files for f in ['pytorch_model.bin', 'model.safetensors', 'model.bin']):
                print(f"[INFO] Found model files in: {root}")
                return root
    return base_path

def load_models_background():
    global siglip_model, siglip_processor, xlm_tokenizer, xlm_model, models_ready
    print("[INFO] Background thread starting model loading...")
    
    available_mem = get_available_memory_mb()
    print(f"[INFO] Available memory: {available_mem:.0f} MB")
    
    if not torch:
        models_ready["status"] = "partial"
        return
    
    if available_mem < 1500:
        print("[WARN] Low memory detected. Skipping local models (cloud-only mode).")
        models_ready["status"] = "cloud-only"
        return
        
    try:
        from transformers import (
            SiglipProcessor, SiglipForImageClassification,
            XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        )
        
        if siglip_model is None:
            try:
                print(f"[INFO] SigLIP base path: {SIGLIP_PATH}")
                print(f"[INFO] SigLIP path exists: {os.path.exists(SIGLIP_PATH)}")
                if os.path.exists(SIGLIP_PATH):
                    print(f"[INFO] SigLIP contents: {os.listdir(SIGLIP_PATH)}")
                siglip_actual_path = find_model_dir(SIGLIP_PATH)
                print(f"[INFO] SigLIP actual path: {siglip_actual_path}")
                if siglip_actual_path and os.path.exists(siglip_actual_path):
                    print(f"[INFO] Loading SIGLIP from: {siglip_actual_path}")
                    siglip_model = SiglipForImageClassification.from_pretrained(
                        siglip_actual_path, 
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32
                    )
                    siglip_processor = SiglipProcessor.from_pretrained(siglip_actual_path)
                else:
                    print(f"[WARN] SIGLIP model files not found in: {SIGLIP_PATH}")
                    raise FileNotFoundError(f"SigLIP model not found at {SIGLIP_PATH}")
                siglip_model = siglip_model.to(device)
                siglip_model.eval()
                print("[OK] SIGLIP model loaded")
                
                dummy_img = Image.new('RGB', (384, 384))
                with torch.no_grad():
                    inputs = siglip_processor(images=dummy_img, return_tensors="pt").to(device)
                    _ = siglip_model(**inputs)
                print("[OK] SIGLIP model warmed up")
            except Exception as e:
                print(f"[FAIL] SIGLIP error: {e}")
                siglip_model = None
        
        if xlm_model is None:
            try:
                print(f"[INFO] XLM base path: {XLM_PATH}")
                print(f"[INFO] XLM path exists: {os.path.exists(XLM_PATH)}")
                if os.path.exists(XLM_PATH):
                    print(f"[INFO] XLM contents: {os.listdir(XLM_PATH)}")
                xlm_actual_path = find_model_dir(XLM_PATH)
                print(f"[INFO] XLM actual path: {xlm_actual_path}")
                if xlm_actual_path and os.path.exists(xlm_actual_path):
                    print(f"[INFO] Loading XLM-RoBERTa from: {xlm_actual_path}")
                    xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_actual_path)
                    xlm_model = XLMRobertaForSequenceClassification.from_pretrained(
                        xlm_actual_path, 
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32
                    )
                else:
                    print(f"[WARN] XLM-RoBERTa model files not found in: {XLM_PATH}")
                    raise FileNotFoundError(f"XLM-RoBERTa model not found at {XLM_PATH}")
                xlm_model = xlm_model.to(device)
                xlm_model.eval()
                print("[OK] XLM-RoBERTa model loaded")
                
                with torch.no_grad():
                    inputs = xlm_tokenizer("warmup", return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    _ = xlm_model(**inputs)
                print("[OK] XLM-RoBERTa model warmed up")
            except Exception as e:
                print(f"[FAIL] XLM-R error: {e}")
                xlm_model = None
            
    except Exception as e:
        print(f"[FAIL] Model loading error: {e}")
        
    models_ready["siglip"] = siglip_model is not None
    models_ready["xlm"] = xlm_model is not None
    models_ready["status"] = "ready"
    print(f"[INFO] Models ready: {models_ready}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    yield

app = FastAPI(title="TruthGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "TruthGuard Rumor Detection API",
        "status": "online",
        "models_status": models_ready["status"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_status": models_ready["status"],
        "siglip_loaded": siglip_model is not None,
        "xlm_loaded": xlm_model is not None,
        "gemini_available": genai_client is not None,
        "tavily_available": tavily_client is not None
    }

@app.get("/models-status")
async def models_status():
    return models_ready

@app.get("/gemini-models")
async def list_gemini_models():
    if not genai_client:
        return {"error": "Gemini not configured"}
    try:
        models = genai_client.list_models()
        return {"models": [m.name for m in models]}
    except Exception as e:
        return {"error": str(e)}

def parse_visual_label(raw_label):
    label_lower = str(raw_label).lower()
    # Explicit mapping for common model outputs
    if "label_0" in label_lower or label_lower == "0" or "fake" in label_lower:
        return "RUMOR"
    if "label_1" in label_lower or label_lower == "1" or "true" in label_lower or "real" in label_lower:
        return "NON-RUMOR"
    
    # Keyword fallback
    if any(x in label_lower for x in ["not_rumor", "non_rumor", "real", "true", "legit"]):
        return "NON-RUMOR"
    if any(x in label_lower for x in ["fake", "false", "rumor", "misinformation", "label_0"]):
        return "RUMOR"
    return "UNKNOWN"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def gemini_generate_retry(model, content, config=None):
    return model.generate_content(content, generation_config=config)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    MAX_SIZE = 10 * 1024 * 1024
    if file.size and file.size > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image too large. Max 10MB allowed.")
    
    if models_ready["status"] == "loading":
        raise HTTPException(status_code=503, detail="Models are still loading in the background. Please try again in 1-2 minutes.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        file.file.seek(0)
        img_bytes = file.file.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        
        results = {
            "visual": "UNAVAILABLE",
            "text": "UNAVAILABLE",
            "tavily": "UNAVAILABLE",
            "gemini": "UNAVAILABLE",
            "final": "UNAVAILABLE",
            "translated": "",
            "original_text": "",
            "sources": [],
            "mode": "full",
            "confidence": 0.0,
            "tavily_analysis": "",
            "gemini_analysis": "",
            "claim_verdict": "NON-RUMOR",
            "report_verdict": "NON-RUMOR"
        }
        
        # 🟢 PILLAR 1: SigLIP (LOCAL)
        visual_display = "UNAVAILABLE"
        if torch and siglip_model and siglip_processor:
            try:
                inputs = siglip_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = siglip_model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax().item()
                
                raw_vis = siglip_model.config.id2label[pred] if hasattr(siglip_model.config, 'id2label') else str(pred)
                visual_display = "RUMOR" if any(x in raw_vis.lower() for x in ["fake", "false", "rumor"]) else "NON-RUMOR"
                results["visual"] = visual_display
                results["visual_confidence"] = float(probs.max().item())
                print(f"[SigLIP] {visual_display}")
            except Exception as e:
                print(f"SigLIP Error: {e}")
        
        # 🔵 PILLAR 3, 4: GEMINI WITH GOOGLE SEARCH GROUNDING
        gemini_verdict = "UNAVAILABLE"
        claim_verdict = "NON-RUMOR"
        report_verdict = "NON-RUMOR"
        
        if genai_client:
            try:
                print("[Gemini] Running analysis with Google Search Grounding...")
                
                model_names = ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-2.5-flash"]
                model = None
                selected_model = None
                
                for model_name in model_names:
                    try:
                        model = genai_client.GenerativeModel(model_name)
                        response = gemini_generate_retry(model, "test", config={"max_output_tokens": 10, "temperature": 0})
                        selected_model = model_name
                        print(f"[Gemini] Using model: {model_name}")
                        break
                    except Exception:
                        continue
                
                if not selected_model:
                    raise Exception("No working Gemini model found")
                
                fact_check_prompt = """You are a professional fact-checker analyzing a news image.

TASK 1: Extract any visible text from the image and translate non-English text to English.

TASK 2: Evaluate the image for signs of manipulation:
- AI-generated artifacts
- Metadata inconsistencies  
- Unusual lighting/shadows
- Poor image quality in specific areas
- Watermark anomalies

TASK 3: Evaluate the CLAIM (text content):
- Is the claim verifiable?
- Does it match official sources?
- Is it misleading or taken out of context?

Return your analysis in this format:

**CLAIM VERDICT:** [TRUE/FALSE/MISLEADING/UNVERIFIABLE]
Explain the claim evaluation.

**REPORT AUTHENTICITY:** [AUTHENTIC/MANIPULATED/SUSPICIOUS]
Explain the image authenticity evaluation.

**DETAILED ANALYSIS:**
[Your comprehensive analysis]"""
                
                response = model.generate_content(
                    [{"mime_type": "image/jpeg", "data": img_base64}, fact_check_prompt],
                    generation_config={"temperature": 0}
                )
                
                response_text = response.text
                
                claim_verdict = "NON-RUMOR"
                report_verdict = "NON-RUMOR"
                
                lines = response_text.split('\n')
                for line in lines:
                    if 'CLAIM VERDICT:' in line.upper():
                        claim_text = line.split('CLAIM VERDICT:')[1].strip().upper()
                        if 'FALSE' in claim_text or 'FAKE' in claim_text or 'MISLEADING' in claim_text:
                            claim_verdict = "RUMOR"
                        elif 'TRUE' in claim_text or 'VERIFIED' in claim_text:
                            claim_verdict = "NON-RUMOR"
                    if 'REPORT AUTHENTICITY:' in line.upper():
                        auth_text = line.split('REPORT AUTHENTICITY:')[1].strip().upper()
                        if 'MANIPULATED' in auth_text or 'SUSPICIOUS' in auth_text or 'FAKE' in auth_text:
                            report_verdict = "RUMOR"
                        elif 'AUTHENTIC' in auth_text or 'GENUINE' in auth_text:
                            report_verdict = "NON-RUMOR"
                
                gemini_verdict = "RUMOR" if (claim_verdict == "RUMOR" or report_verdict == "RUMOR") else "NON-RUMOR"
                
                grounding_chunks = []
                try:
                    if hasattr(response, 'grounding_metadata') and response.grounding_metadata:
                        chunks = response.grounding_metadata.get('grounding_chunks', [])
                        for chunk in chunks[:5]:
                            web = chunk.get('web', {})
                            if web.get('uri'):
                                grounding_chunks.append({
                                    "title": web.get('title', 'Source'),
                                    "url": web.get('uri')
                                })
                except Exception as e:
                    print(f"[Gemini] Grounding metadata error: {e}")
                
                results["sources"] = grounding_chunks
                results["gemini"] = gemini_verdict
                results["gemini_analysis"] = response_text
                results["gemini_model_used"] = selected_model
                results["claim_verdict"] = claim_verdict
                results["report_verdict"] = report_verdict
                
                print(f"[Gemini] Claim: {claim_verdict}, Report: {report_verdict} -> Final: {gemini_verdict}")
                
            except Exception as e:
                print(f"Gemini Error: {e}")
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str:
                    results["gemini"] = "QUOTA EXCEEDED"
                    results["gemini_analysis"] = "Google Gemini API quota exceeded."
                else:
                    results["gemini"] = "ERROR"
                    results["gemini_analysis"] = f"Failed: {str(e)}"
                gemini_verdict = "UNAVAILABLE"

        # 🔵 PILLAR 2: XLM-R (LOCAL - TEXT CLASSIFICATION) - INDEPENDENT
        extracted_text = ""
        text_display = "UNAVAILABLE"
        
        if OCR_AVAILABLE:
            try:
                extracted_text = pytesseract.image_to_string(image)
                extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()[:500]
                print(f"[OCR] Extracted: {extracted_text[:100]}...")
            except Exception as e:
                print(f"[OCR] Error: {e}")
                extracted_text = ""
        
        text_input = extracted_text if extracted_text else "This is a news article containing information about current events and public announcements"
        
        if torch and xlm_model and xlm_tokenizer:
            try:
                inputs_xlm = xlm_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                with torch.no_grad():
                    logits = xlm_model(**inputs_xlm).logits
                    text_probs = torch.softmax(logits, dim=1)
                    text_pred = text_probs.argmax().item()
                    confidence = text_probs.max().item()

                raw_txt = xlm_model.config.id2label[text_pred] if hasattr(xlm_model.config, 'id2label') else str(text_pred)
                
                if "0" in raw_txt or "fake" in raw_txt.lower() or "rumor" in raw_txt.lower() or "false" in raw_txt.lower():
                    text_display = "RUMOR"
                elif "1" in raw_txt or "real" in raw_txt.lower() or "true" in raw_txt.lower() or "non_rumor" in raw_txt.lower():
                    text_display = "NON-RUMOR"
                else:
                    text_display = "NON-RUMOR"
                
                results["text"] = text_display
                results["text_confidence"] = float(confidence)
                print(f"[XLM-R] Label:{raw_txt} -> {text_display} (conf:{confidence:.2%})")
            except Exception as e:
                print(f"XLM-R Error: {e}")
                results["text"] = "ERROR"
                text_display = "ERROR"

        # 📑 PILLAR 3: TAVILY (WEB RESEARCH) - INDEPENDENT
        search_command = extracted_text[:200] if extracted_text else "latest news verified facts"
        verdict_tav = "UNAVAILABLE"
        
        if tavily_client:
            try:
                cache_key = search_command.lower().strip()
                if cache_key in tavily_cache:
                    print(f"[Tavily] Using cached result for: {search_command[:80]}...")
                    tav_res = tavily_cache[cache_key]
                else:
                    print(f"[Tavily] Searching: {search_command[:80]}...")
                    tav_res = tavily_client.search(
                        query=search_command, 
                        search_depth="advanced", 
                        include_answer="advanced", 
                        max_results=5
                    )
                    tavily_cache[cache_key] = tav_res
                
                tav_sources = [{"title": r.get("title"), "url": r.get("url")} for r in tav_res.get("results", [])]
                if not results["sources"]:
                    results["sources"] = tav_sources
                
                tav_answer = tav_res.get("answer", "")
                tav_context = " ".join([r.get("content", "") for r in tav_res.get("results", [])[:3]])
                
                combined_text = (tav_answer + " " + tav_context).lower()
                
                strong_rumor = ["false claim", "fake news", "hoax", "misinformation", "debunked", "fabricated", 
                               "manipulated image", "ai-generated", "outdated", "satire presented as news"]
                weak_rumor = ["misleading", "unverified", "no evidence", "cannot verify", "unconfirmed", "clickbait"]
                strong_fact = ["confirmed by", "official statement", "verified", "true", "according to official",
                              "government said", "police confirmed", "hospital confirmed", "authorities say"]
                weak_fact = ["reported", "according to", "sources say", "apparently", "unclear"]
                
                rumor_score = 0
                fact_score = 0
                
                for phrase in strong_rumor:
                    if phrase in combined_text:
                        rumor_score += 2
                        
                for phrase in weak_rumor:
                    if phrase in combined_text:
                        rumor_score += 1
                        
                for phrase in strong_fact:
                    if phrase in combined_text:
                        fact_score += 2
                        
                for phrase in weak_fact:
                    if phrase in combined_text:
                        fact_score += 1
                
                if not tav_answer or len(tav_answer) < 15 or "no relevant" in tav_answer.lower():
                    verdict_tav = "NON-RUMOR"
                elif rumor_score > fact_score + 1:
                    verdict_tav = "RUMOR"
                elif fact_score > rumor_score + 1:
                    verdict_tav = "NON-RUMOR"
                elif rumor_score > fact_score:
                    verdict_tav = "RUMOR"
                elif rumor_score == fact_score and rumor_score > 0:
                    verdict_tav = "NON-RUMOR"
                else:
                    verdict_tav = "NON-RUMOR"
                
                results["tavily"] = verdict_tav
                results["tavily_analysis"] = tav_answer if tav_answer else "Web search completed. Review sources."
                print(f"[Tavily] R-score:{rumor_score} F-score:{fact_score} -> {verdict_tav}")
            except Exception as e:
                print(f"Tavily Error: {e}")
                results["tavily"] = "ERROR"
                results["tavily_analysis"] = f"Search failed: {str(e)}"
                verdict_tav = "ERROR"

        # 🗳️ FINAL MAJORITY DECISION - RESPECTING INDIVIDUAL MODEL OUTPUTS
        verdicts = {
            "visual": visual_display,
            "text": text_display,
            "tavily": verdict_tav,
            "gemini": gemini_verdict
        }
        
        valid_votes = [v for v in verdicts.values() if v in ["RUMOR", "NON-RUMOR"]]
        r_count = valid_votes.count("RUMOR")
        nr_count = valid_votes.count("NON-RUMOR")
        total_valid = len(valid_votes)

        if r_count > nr_count:
            majority_verdict = "RUMOR"
        elif nr_count > r_count:
            majority_verdict = "NON-RUMOR"
        else:
            majority_verdict = text_display if text_display in ["RUMOR", "NON-RUMOR"] else "NON-RUMOR"

        results["final"] = majority_verdict
        results["confidence"] = float(max(r_count, nr_count) / total_valid) if total_valid > 0 else 0.5

        print(f"[FINAL] R:{r_count} NR:{nr_count} -> {majority_verdict}")
        return results
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
