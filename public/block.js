// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    // Parse URL parameters to get details
    const urlParams = new URLSearchParams(window.location.search);
    const maliciousUrl = urlParams.get('url');
    const reason = urlParams.get('reason');
    const conf = urlParams.get('confidence');
    const goBackUrl = urlParams.get('goBack') || '';

    // Populate UI
    if (reason) {
        document.getElementById('reason').textContent = `Reason: ${reason}`;
    }

    if (conf) {
        const confPercent = (parseFloat(conf) * 100).toFixed(1);
        document.getElementById('confidence').textContent = `Confidence: ${confPercent}%`;
    }

    // Handle "Go Back" — tell the background script to navigate the tab
    // Using chrome.tabs.update from background avoids re-triggering the block
    document.getElementById('go-back').addEventListener('click', () => {
        chrome.runtime.sendMessage({
            action: 'GO_BACK',
            goBackUrl: goBackUrl || 'chrome://newtab'
        }, () => {
            // Fallback: if sendMessage fails or goBackUrl is empty, open new tab
            if (chrome.runtime.lastError || !goBackUrl) {
                window.open('chrome://newtab', '_self');
            }
        });
    });

    // Handle "Proceed (Unsafe)"
    document.getElementById('proceed').addEventListener('click', () => {
        if (maliciousUrl) {
            const decodedUrl = decodeURIComponent(maliciousUrl);
            chrome.runtime.sendMessage({
                action: 'ALLOW_URL',
                url: decodedUrl
            }, () => {
                window.location.href = decodedUrl;
            });
        }
    });
});
