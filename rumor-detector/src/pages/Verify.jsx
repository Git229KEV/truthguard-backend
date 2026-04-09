import { useState, useCallback, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const Verify = () => {
  const [selectedImage, setSelectedImage] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processStep, setProcessStep] = useState(0)
  const [result, setResult] = useState(null)
  const [apiStatus, setApiStatus] = useState(null)
  const [detailedResult, setDetailedResult] = useState(null)
  const [loadingModels, setLoadingModels] = useState([])
  const [clipboardError, setClipboardError] = useState(null)

  useEffect(() => {
    checkApiStatus()
    const interval = setInterval(checkApiStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`)
      if (response.ok) {
        const data = await response.json()
        setApiStatus(data)
        updateLoadingModels(data)
      }
    } catch (error) {
      setApiStatus({ status: 'offline', error: error.message })
      setLoadingModels([])
    }
  }

  const updateLoadingModels = (data) => {
    const models = []
    
    if (!data.siglip_loaded) {
      models.push({ name: 'SigLIP Vision', status: 'loading', icon: '👁️' })
    } else {
      models.push({ name: 'SigLIP Vision', status: 'ready', icon: '👁️' })
    }
    
    if (!data.xlm_loaded) {
      models.push({ name: 'XLM-RoBERTa', status: 'loading', icon: '🔤' })
    } else {
      models.push({ name: 'XLM-RoBERTa', status: 'ready', icon: '🔤' })
    }
    
    if (!data.gemini_available) {
      models.push({ name: 'Gemini AI', status: 'loading', icon: '🤖' })
    } else {
      models.push({ name: 'Gemini AI', status: 'ready', icon: '🤖' })
    }
    
    if (!data.tavily_available) {
      models.push({ name: 'Tavily Search', status: 'loading', icon: '🌐' })
    } else {
      models.push({ name: 'Tavily Search', status: 'ready', icon: '🌐' })
    }
    
    setLoadingModels(models)
  }

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target.result)
        setResult(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target.result)
        setResult(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const processImage = async () => {
    setIsProcessing(true)
    setProcessStep(1)
    setDetailedResult(null)
    
    try {
      const response = await fetch(selectedImage)
      const blob = await response.blob()
      const formData = new FormData()
      formData.append('file', blob, 'news-image.jpg')
      
      setProcessStep(2)
      
      const apiResponse = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
      })
      
      setProcessStep(3)
      
      if (!apiResponse.ok) {
        throw new Error('Analysis failed')
      }
      
      setProcessStep(4)
      const data = await apiResponse.json()
      
      setProcessStep(5)
      setIsProcessing(false)
      
      setDetailedResult(data)
      setResult(data.final === 'RUMOR' ? 'rumor' : 'legitimate')
      
    } catch (error) {
      console.error('Error analyzing image:', error)
      setIsProcessing(false)
      setResult('error')
      setDetailedResult({ error: error.message || 'Analysis failed. Please try again.' })
    }
  }

  const resetUpload = () => {
    setSelectedImage(null)
    setIsProcessing(false)
    setProcessStep(0)
    setResult(null)
  }

  const processSteps = [
    { id: 1, label: 'Uploading image', icon: '📤' },
    { id: 2, label: 'Extracting text & visual features', icon: '🔍' },
    { id: 3, label: 'Analyzing content patterns', icon: '🧠' },
    { id: 4, label: 'Cross-checking with sources', icon: '🌐' },
    { id: 5, label: 'Generating verification report', icon: '📊' }
  ]

  return (
    <div className="verify-page">
      <div className="verify-container">
        <div className="verify-header">
          <h1 className="verify-title">Verify News Image</h1>
          <p className="verify-subtitle">
            Upload a news screenshot or image and we'll analyze it for potential misinformation
          </p>
          {apiStatus && (
            <div className={`api-status ${apiStatus.models_status === 'ready' ? 'ready' : 'loading'}`}>
              <span className="status-dot"></span>
              <span>Backend: {
                apiStatus.models_status === 'ready' ? 'Full Mode' :
                apiStatus.models_status === 'partial' ? 'Limited Mode' :
                apiStatus.models_status === 'cloud-only' ? 'Cloud Mode' :
                apiStatus.models_status
              }</span>
            </div>
          )}
        </div>

        {!selectedImage ? (
          <div 
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="upload-content">
              <div className="upload-icon-large">🖼️</div>
              <h3 className="upload-title">Drop your news image here</h3>
              <p className="upload-subtitle">or click to browse files</p>
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleFileSelect}
                className="file-input"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="btn btn-primary">
                Choose Image
              </label>
              <p className="upload-note">Supports PNG, JPG, WEBP up to 10MB</p>
            </div>
          </div>
        ) : (
          <div className="analysis-container">
            <div className="image-preview-section">
              <div className="image-preview">
                <img src={selectedImage} alt="Selected news" />
                {!isProcessing && !result && (
                  <div className="image-overlay">
                    <button className="btn btn-secondary small" onClick={resetUpload}>
                      Change Image
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="processing-section">
              {!isProcessing && !result && (
                <div className="action-buttons">
                  <button className="btn btn-primary large" onClick={processImage}>
                    🔍 Analyze Image
                  </button>
                </div>
              )}

              {isProcessing && (
                <div className="processing-steps">
                  <h3 className="processing-title">Analyzing your image...</h3>
                  <div className="steps-list">
                    {processSteps.map((step) => (
                      <div 
                        key={step.id} 
                        className={`step-item ${
                          processStep > step.id ? 'completed' : 
                          processStep === step.id ? 'active' : 'pending'
                        }`}
                      >
                        <div className="step-icon">
                          {processStep > step.id ? '✓' : step.icon}
                        </div>
                        <div className="step-info">
                          <span className="step-label">{step.label}</span>
                          {processStep === step.id && (
                            <span className="step-spinner"></span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result && result !== 'error' && (
                <div className={`result-card ${result}`}>
                  <div className="result-icon-large">
                    {result === 'rumor' ? '⚠️' : '✅'}
                  </div>
                  <h2 className="result-title">
                    {result === 'rumor' ? 'Potential Rumor Detected' : 'Likely Legitimate'}
                  </h2>
                  <p className="result-description">
                    {result === 'rumor' 
                      ? 'Our analysis indicates this content may contain misleading or false information. Exercise caution before sharing.'
                      : 'This content appears to be legitimate based on our verification checks.'
                    }
                  </p>
                  {detailedResult && (
                    <div className="detailed-analysis">
                      <h4>Analysis Breakdown</h4>
                      <div className="analysis-items">
                        <div className={`analysis-item ${detailedResult.visual.toLowerCase().replace('-', '')}`}>
                          <span className="item-label">Visual Analysis:</span>
                          <span className="item-value">{detailedResult.visual}</span>
                          {detailedResult.visual_confidence && (
                            <span className="item-confidence">({(detailedResult.visual_confidence * 100).toFixed(0)}%)</span>
                          )}
                        </div>

                        <div className={`analysis-item ${detailedResult.text.toLowerCase().replace('-', '')}`}>
                          <span className="item-label">Text Analysis:</span>
                          <span className="item-value">{detailedResult.text}</span>
                          {detailedResult.text_confidence && (
                            <span className="item-confidence">({(detailedResult.text_confidence * 100).toFixed(0)}%)</span>
                          )}
                        </div>

                        <div className={`analysis-item ${detailedResult.gemini.toLowerCase().replace('-', '')}`}>
                          <span className="item-label">Forensic Check:</span>
                          <span className="item-value">{detailedResult.gemini}</span>
                          {detailedResult.gemini_analysis && (
                            <div className="analysis-reasoning">
                              <span className="reasoning-label">AI Forensic Report:</span>
                              {detailedResult.gemini_analysis}
                            </div>
                          )}
                        </div>

                        <div className={`analysis-item ${detailedResult.tavily.toLowerCase().replace('-', '')}`}>
                          <span className="item-label">Web Verification:</span>
                          <span className="item-value">{detailedResult.tavily}</span>
                        </div>
                      </div>
                      <div className="confidence-bar">
                        <span>Confidence: {(detailedResult.confidence * 100).toFixed(0)}%</span>
                        <div className="bar">
                          <div className="fill" style={{width: `${detailedResult.confidence * 100}%`}}></div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="result-actions">
                    <button className="btn btn-primary" onClick={resetUpload}>
                      Verify Another Image
                    </button>
                  </div>
                </div>
              )}
              
              {result === 'error' && detailedResult && (
                <div className="result-card error">
                  <div className="result-icon-large">❌</div>
                  <h2 className="result-title">Analysis Failed</h2>
                  <p className="result-description">
                    {detailedResult.error || 'An error occurred during analysis. Please try again.'}
                  </p>
                  <div className="result-actions">
                    <button className="btn btn-primary" onClick={resetUpload}>
                      Try Again
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Verify
