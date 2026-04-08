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
if not SIGLIP_PATH or not os.path.exists(SIGLIP_PATH):
    SIGLIP_PATH = default_siglip_path

XLM_PATH = os.getenv("XLM_ROBERTA_MODEL_PATH")
if XLM_PATH:
    XLM_PATH = XLM_PATH.strip('"').strip("'")
if not XLM_PATH or not os.path.exists(XLM_PATH):
    XLM_PATH = default_xlm_path

print(f"[INFO] SIGLIP_PATH: {SIGLIP_PATH}")
print(f"[INFO] XLM_PATH: {XLM_PATH}")

genai_client = None
tavily_client = None

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

def load_models_background():
    global siglip_model, siglip_processor, xlm_tokenizer, xlm_model, models_ready
    print("[INFO] Background thread starting model loading...")
    if not torch:
        models_ready["status"] = "partial"
        return
        
    try:
        from transformers import (
            SiglipProcessor, SiglipForImageClassification,
            XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        )
        
        if SIGLIP_PATH and os.path.exists(SIGLIP_PATH):
            print(f"[INFO] Loading SIGLIP from: {SIGLIP_PATH}")
            siglip_model = SiglipForImageClassification.from_pretrained(SIGLIP_PATH, low_cpu_mem_usage=True)
            siglip_model = siglip_model.to(device)
            siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_PATH)
            siglip_model.eval()
            print("[OK] SIGLIP model loaded")
            
            dummy_img = Image.new('RGB', (224, 224))
            with torch.no_grad():
                inputs = siglip_processor(images=dummy_img, return_tensors="pt").to(device)
                _ = siglip_model(**inputs)
            print("[OK] SIGLIP model warmed up")
        else:
            print(f"[WARN] SIGLIP path not found: {SIGLIP_PATH}")
        
        if XLM_PATH and os.path.exists(XLM_PATH):
            print(f"[INFO] Loading XLM-RoBERTa from: {XLM_PATH}")
            xlm_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_PATH)
            xlm_model = XLMRobertaForSequenceClassification.from_pretrained(XLM_PATH, low_cpu_mem_usage=True)
            xlm_model = xlm_model.to(device)
            xlm_model.eval()
            print("[OK] XLM-RoBERTa model loaded")
            
            with torch.no_grad():
                inputs = xlm_tokenizer("warmup", return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                _ = xlm_model(**inputs)
            print("[OK] XLM-RoBERTa model warmed up")
        else:
            print(f"[WARN] XLM-RoBERTa path not found: {XLM_PATH}")
            
    except Exception as e:
        print(f"[FAIL] Model loading error: {e}")
        
    models_ready["siglip"] = siglip_model is not None
    models_ready["xlm"] = xlm_model is not None
    models_ready["status"] = "ready" if siglip_model else "partial"
    print(f"[INFO] Models ready: {models_ready}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    yield

app = FastAPI(title="TruthGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174"],
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
                        response = gemini_generate_retry(model, "test", config={"max_output_tokens": 10})
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
                    generation_config={"temperature": 0.1}
                )
                
                response_text = response.text
                
                lines = response_text.split('\n')
                for line in lines:
                    if 'CLAIM VERDICT:' in line.upper():
                        claim_text = line.split('CLAIM VERDICT:')[1].strip().upper()
                        if 'FALSE' in claim_text or 'FAKE' in claim_text or 'MISLEADING' in claim_text:
                            claim_verdict = "RUMOR"
                        else:
                            claim_verdict = "NON-RUMOR"
                    if 'REPORT AUTHENTICITY:' in line.upper():
                        auth_text = line.split('REPORT AUTHENTICITY:')[1].strip().upper()
                        if 'MANIPULATED' in auth_text or 'SUSPICIOUS' in auth_text or 'FAKE' in auth_text:
                            report_verdict = "RUMOR"
                        else:
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
                print(f"[Tavily] Searching: {search_command[:80]}...")
                tav_res = tavily_client.search(
                    query=search_command, 
                    search_depth="advanced", 
                    include_answer="advanced", 
                    max_results=5
                )
                
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
