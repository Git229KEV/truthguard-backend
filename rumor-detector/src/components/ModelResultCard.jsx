import React from 'react';

const ShieldIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);

const WarningIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const CheckIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const BrainIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A5 5 0 0 1 12 7.3a5 5 0 0 1 2.5-5.3 5 5 0 0 1 0 10.5 5 5 0 0 1-5 0 5 5 0 0 1 0-10.5z" />
    <path d="M12 12v10" />
    <path d="M8 22h8" />
  </svg>
);

const SearchIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const EyeIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const TypeIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="4 7 4 4 20 4 20 7" />
    <line x1="9" y1="20" x2="15" y2="20" />
    <line x1="12" y1="4" x2="12" y2="20" />
  </svg>
);

const ModelResultCard = ({ 
  modelId, 
  modelName, 
  variant, // 'gemini', 'tavily', 'siglip', 'xlm'
  verdict, // 'RUMOR' or 'NON-RUMOR'
  reasoning,
  confidence,
  extraInfo
}) => {
  const isRumor = verdict === 'RUMOR';
  
  const getIcon = () => {
    switch (variant) {
      case 'gemini': return <BrainIcon />;
      case 'tavily': return <SearchIcon />;
      case 'siglip': return <EyeIcon />;
      case 'xlm': return <TypeIcon />;
      default: return <ShieldIcon />;
    }
  };

  return (
    <div className="model-result-card">
      <div className={`model-header-tag ${variant}`}>
        {getIcon()}
        <span>MODEL {modelId}: {modelName}</span>
      </div>

      <div className="card-title-row">
        <ShieldIcon />
        <span>Analysis Report</span>
      </div>

      <div className={`verdict-badge-large ${isRumor ? 'rumor' : 'legitimate'}`}>
        {isRumor ? <WarningIcon /> : <CheckIcon />}
        <span>{isRumor ? 'FAKE / RUMOR' : 'LIKELY LEGITIMATE'}</span>
      </div>

      <div className="analysis-report-container">
        <div className="report-header">
          <ShieldIcon />
          <span>{variant === 'gemini' ? 'DEEP ANALYSIS' : 'ANALYSIS BREAKDOWN'}</span>
        </div>
        
        {extraInfo && (
          <span className="report-subtitle">{extraInfo}</span>
        )}
        
        <div className="report-content">
          {reasoning || "Analysis details are being compiled..."}
        </div>
        
        {confidence && (
          <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: '#64748b', fontWeight: 'bold' }}>
            CONFIDENCE SCORE: {(confidence * 100).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelResultCard;
