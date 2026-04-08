# TruthGuard - News Rumor Detection Dashboard

A modern web application for detecting rumors in news images using AI.

## 🚀 Features

### Frontend (React + Vite)
- **Homepage**: Modern hero section, features grid, and "Why Choose Us" section
- **Verify Page**: Drag-and-drop image upload, processing visualization, and results display
- **Polished UI**: Dark theme, glassmorphism effects, smooth animations, responsive design

### Backend (FastAPI)
- Image analysis pipeline with multiple AI models
- SIGLIP visual analysis
- XLM-Roberta text classification
- Google Gemini OCR and forensic analysis
- Tavily search cross-verification
- Multi-model voting system for final verdict

## 📦 Tech Stack

**Frontend:**
- React 18
- Vite
- React Router DOM
- Modern CSS with gradients and animations

**Backend:**
- FastAPI
- PyTorch
- Hugging Face Transformers
- Google Generative AI
- Tavily Python
- Pillow

## 🛠️ Installation

### Prerequisites
- Node.js 18+
- Python 3.10+
- API keys for Google Gemini and Tavily

### Frontend Setup
```bash
cd rumor-detector
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env`
2. Add your API keys
3. Run the server:
```bash
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`

## 📁 Project Structure

```
Major Project/
├── rumor-detector/          # React frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
│   └── package.json
├── backend/                 # FastAPI backend
│   ├── models/             # ML models directory
│   ├── main.py
│   ├── requirements.txt
│   └── .env
└── README.md
```

## 🔑 API Keys Required

- **Google Gemini**: Get from [Google AI Studio](https://makersuite.google.com/)
- **Tavily**: Get from [Tavily](https://tavily.com/)

## 🎯 Usage

1. Start both frontend and backend servers
2. Open `http://localhost:5173`
3. Go to "Verify News" page
4. Upload a news image or screenshot
5. View the analysis results and final verdict

## 📝 Notes

- ML models need to be placed in `backend/models/` directory
- Backend will work in demo mode without models for testing UI
- For full functionality, ensure models are downloaded and API keys are configured
