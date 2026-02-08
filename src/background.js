
// Background Service Worker

const API_URL = 'http://localhost:5000/predict';

// Function to check URL safety
async function checkUrl(url, tabId) {
    try {
        // Basic validation
        if (!url || !url.startsWith('http')) return;

        // Set badge to "..." while checking
        chrome.action.setBadgeText({ text: '...', tabId });
        chrome.action.setBadgeBackgroundColor({ color: '#888', tabId });

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });

        const data = await response.json();

        if (data.is_dangerous) {
            // DANGEROUS
            chrome.action.setBadgeText({ text: '!', tabId });
            chrome.action.setBadgeBackgroundColor({ color: '#FF0000', tabId });

            // Inject content script if not already present (declarative approach is better but this works for v3)
            // Send message to content script
            chrome.tabs.sendMessage(tabId, {
                action: 'SHOW_WARNING',
                reason: data.reason,
                confidence: data.confidence
            }).catch(err => console.log("Content script might not be ready:", err));

        } else {
            // SAFE
            chrome.action.setBadgeText({ text: 'OK', tabId });
            chrome.action.setBadgeBackgroundColor({ color: '#00CC00', tabId });
        }

    } catch (error) {
        console.error('Prediction error:', error);
        chrome.action.setBadgeText({ text: 'ERR', tabId });
        chrome.action.setBadgeBackgroundColor({ color: '#000', tabId });
    }
}

// Listen for tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        checkUrl(tab.url, tabId);
    }
});

// Listen for tab activation (switch tabs)
chrome.tabs.onActivated.addListener((activeInfo) => {
    chrome.tabs.get(activeInfo.tabId, (tab) => {
        if (tab.url) {
            checkUrl(tab.url, activeInfo.tabId);
        }
    });
});
