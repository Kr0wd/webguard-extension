import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [currentUrl, setCurrentUrl] = useState('')
  const [loading, setLoading] = useState(true)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Get current tab URL
    if (typeof chrome !== 'undefined' && chrome.tabs) {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
          const url = tabs[0].url
          setCurrentUrl(url)
          checkUrl(url)
        }
      })
    } else {
      // Fallback for development
      setCurrentUrl('http://example.com/test')
      checkUrl('http://example.com/test')
    }
  }, [])

  const checkUrl = async (url) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get prediction')
      }

      setPrediction(data)
    } catch (err) {
      setError(err.message)
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRecheck = () => {
    if (currentUrl) {
      checkUrl(currentUrl)
    }
  }

  return (
    <div className="container">
      <h1>🛡️ URL Safety Checker</h1>
      
      <div className="url-display">
        <strong>Current URL:</strong>
        <div className="url-text">{currentUrl || 'Loading...'}</div>
      </div>

      {loading && (
        <div className="status loading">
          <div className="spinner"></div>
          <p>Analyzing URL...</p>
        </div>
      )}

      {error && (
        <div className="status error">
          <h2>❌ Error</h2>
          <p>{error}</p>
          <p className="hint">Make sure the Flask server is running on http://localhost:5000</p>
          <button onClick={handleRecheck}>Try Again</button>
        </div>
      )}

      {!loading && !error && prediction && (
        <div className={`status ${prediction.is_dangerous ? 'dangerous' : 'safe'}`}>
          <h2>
            {prediction.is_dangerous ? '⚠️ Warning' : '✅ Safe'}
          </h2>
          <p className="verdict">
            {prediction.is_dangerous 
              ? 'This URL appears to be DANGEROUS' 
              : 'This URL appears to be SAFE'}
          </p>
          {prediction.confidence && (
            <p className="confidence">
              Confidence: {(prediction.confidence * 100).toFixed(1)}%
            </p>
          )}
          <button onClick={handleRecheck}>Recheck</button>
        </div>
      )}
    </div>
  )
}

export default App

