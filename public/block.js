// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    // Parse URL parameters to get details
    const urlParams = new URLSearchParams(window.location.search);
    const maliciousUrl = urlParams.get('url');
    const reason = urlParams.get('reason');
    const conf = urlParams.get('confidence');

    // Populate UI
    if (reason) {
        document.getElementById('reason').textContent = `Reason: ${reason}`;
    }

    if (conf) {
        const confPercent = (parseFloat(conf) * 100).toFixed(1);
        document.getElementById('confidence').textContent = `Confidence: ${confPercent}%`;
    }

    // Handle "Go Back"
    document.getElementById('go-back').addEventListener('click', () => {
        // Attempt to go back, if that fails (e.g., opened in new tab), close tab
        if (window.history.length > 1) {
            window.history.back();
        } else {
            window.close();
        }
    });

    // Handle "Proceed (Unsafe)"
    document.getElementById('proceed').addEventListener('click', () => {
        if (maliciousUrl) {
            // Decode and redirect
            const decodedUrl = decodeURIComponent(maliciousUrl);

            // We need to tell the background script to temporarily allow this URL
            chrome.runtime.sendMessage({
                action: 'ALLOW_URL',
                url: decodedUrl
            }, () => {
                // After getting permission, redirect
                window.location.href = decodedUrl;
            });
        }
    });
});
