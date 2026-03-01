
// Content Script

console.log("WebGuard Content Script Loaded");

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === 'SHOW_WARNING') {
    showWarningOverlay(message.reason, message.confidence);
  }
});

function showWarningOverlay(reason, confidence) {
  // Check if overlay already exists
  if (document.getElementById('webguard-warning-overlay')) return;

  const overlay = document.createElement('div');
  overlay.id = 'webguard-warning-overlay';

  // Styles
  overlay.style.position = 'fixed';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.width = '100vw';
  overlay.style.height = '100vh';
  overlay.style.backgroundColor = 'rgba(255, 0, 0, 0.95)';
  overlay.style.backdropFilter = 'blur(10px)'; // Add blur to obscure background
  overlay.style.zIndex = '2147483647'; // Maximum possible z-index
  overlay.style.display = 'flex';
  overlay.style.flexDirection = 'column';
  overlay.style.alignItems = 'center';
  overlay.style.justifyContent = 'center';
  overlay.style.color = '#fff';
  overlay.style.fontFamily = 'system-ui, -apple-system, sans-serif';
  overlay.style.textAlign = 'center';

  // Disable body scroll when warning is present
  document.body.style.overflow = 'hidden';

  const confPercent = confidence ? (confidence * 100).toFixed(1) : '99.9';

  overlay.innerHTML = `
    <h1 style="font-size: 3rem; margin-bottom: 20px;">⚠️ WARNING: Dangerous Site Detected ⚠️</h1>
    <p style="font-size: 1.5rem; margin-bottom: 10px;">Reason: ${reason || 'Unknown'}</p>
    <p style="font-size: 1.2rem; margin-bottom: 30px; opacity: 0.8;">Confidence: ${confPercent}%</p>
    
    <div style="display: flex; gap: 20px;">
      <button id="webguard-go-back" style="
        padding: 15px 30px;
        font-size: 1.2rem;
        background-color: #fff;
        color: #d00;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
      ">Go Back (Recommended)</button>
      
      <button id="webguard-proceed" style="
        padding: 15px 30px;
        font-size: 1.2rem;
        background-color: transparent;
        color: #fff;
        border: 2px solid #fff;
        border-radius: 8px;
        cursor: pointer;
        opacity: 0.7;
      ">Proceed (Unsafe)</button>
    </div>
  `;

  document.body.appendChild(overlay);

  // Event Listeners
  document.getElementById('webguard-go-back').addEventListener('click', () => {
    window.history.back();
  });

  document.getElementById('webguard-proceed').addEventListener('click', () => {
    // Re-enable scrolling when proceeding
    document.body.style.overflow = '';
    overlay.remove();
  });
}
