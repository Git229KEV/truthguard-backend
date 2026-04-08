import { Link } from 'react-router-dom'

const Home = () => {
  const features = [
    {
      icon: '⚡',
      title: 'Lightning Fast',
      description: 'Get instant verification results in seconds with our advanced AI processing pipeline.'
    },
    {
      icon: '🎯',
      title: 'High Accuracy',
      description: '98% accuracy rate using state-of-the-art computer vision and NLP models.'
    },
    {
      icon: '🔒',
      title: 'Privacy First',
      description: 'Your uploaded images are never stored. We process everything securely in real-time.'
    },
    {
      icon: '📱',
      title: 'Easy to Use',
      description: 'Simply upload a news screenshot and get clear, understandable results.'
    }
  ]

  const whyChooseUs = [
    {
      title: 'Advanced Image Analysis',
      description: 'Unlike other tools that only analyze text, we extract and verify information directly from images.',
      icon: '🖼️'
    },
    {
      title: 'Multi-source Cross-checking',
      description: 'We verify claims against thousands of trusted sources and fact-checking databases.',
      icon: '🌐'
    },
    {
      title: 'Real-time Processing',
      description: 'No waiting in queues. Get verification results instantly as soon as you upload.',
      icon: '⏱️'
    },
    {
      title: 'Transparent Explanations',
      description: 'We don\'t just give you a verdict - we show you exactly how we reached our conclusion.',
      icon: '📊'
    }
  ]

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">🚀 Trusted by 50,000+ users</div>
          <h1 className="hero-title">
            Detect Fake News
            <br />
            <span className="gradient-text">In Seconds</span>
          </h1>
          <p className="hero-description">
            Upload any news image or screenshot and let our AI-powered system
            determine if it's real or a rumor. Stop misinformation before it spreads.
          </p>
          <div className="hero-cta">
            <Link to="/verify" className="btn btn-primary">
              Start Verifying Now
            </Link>
            <button className="btn btn-secondary">
              Learn How It Works
            </button>
          </div>
          <div className="hero-stats">
            <div className="stat-item">
              <span className="stat-number">2M+</span>
              <span className="stat-label">Images Verified</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">98%</span>
              <span className="stat-label">Accuracy Rate</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">50K+</span>
              <span className="stat-label">Active Users</span>
            </div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="demo-card">
            <div className="demo-header">
              <div className="demo-dots">
                <span className="dot red"></span>
                <span className="dot yellow"></span>
                <span className="dot green"></span>
              </div>
              <span className="demo-title">TruthGuard Analysis</span>
            </div>
            <div className="demo-content">
              <div className="demo-image-placeholder">
                <span className="upload-icon">📷</span>
                <span>Drop news image here</span>
              </div>
              <div className="demo-result rumor">
                <span className="result-icon">⚠️</span>
                <span className="result-text">Potential Rumor Detected</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="section-header">
          <h2 className="section-title">Powerful Features</h2>
          <p className="section-subtitle">
            Everything you need to verify news authenticity with confidence
          </p>
        </div>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <div className="feature-icon">{feature.icon}</div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Why Choose Us Section */}
      <section className="why-section">
        <div className="section-header">
          <h2 className="section-title">Why Choose TruthGuard?</h2>
          <p className="section-subtitle">
            What makes us different from other verification tools
          </p>
        </div>
        <div className="why-grid">
          {whyChooseUs.map((item, index) => (
            <div key={index} className="why-card">
              <div className="why-icon">{item.icon}</div>
              <h3 className="why-title">{item.title}</h3>
              <p className="why-description">{item.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2 className="cta-title">Ready to Fight Misinformation?</h2>
          <p className="cta-description">
            Start verifying news images today and help stop the spread of rumors.
          </p>
          <Link to="/verify" className="btn btn-primary btn-large">
            Start Verifying Now →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="logo-icon">🔍</span>
            <span className="logo-text">TruthGuard</span>
            <p className="footer-tagline">Verifying news, one image at a time.</p>
          </div>
          <div className="footer-bottom">
            <p>© 2026 TruthGuard. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Home
