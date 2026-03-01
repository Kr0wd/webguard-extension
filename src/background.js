// Background Service Worker

const API_URL = 'http://localhost:5000/predict';

// Set of temporarily allowed URLs (so we don't block them again if the user clicks "Proceed")
const allowedUrls = new Set();

// Function to check URL safety
async function checkUrl(url, tabId) {
    try {
        // Basic validation
        if (!url || !url.startsWith('http')) return;

        // Check if the user has explicitly bypassed the warning for this URL
        if (allowedUrls.has(url)) return;

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

            // Redirect the tab to our local block page instead of injecting content script
            const blockUrl = chrome.runtime.getURL('block.html') +
                `?url=${encodeURIComponent(url)}` +
                `&reason=${encodeURIComponent(data.reason)}` +
                `&confidence=${encodeURIComponent(data.confidence)}`;

            chrome.tabs.update(tabId, { url: blockUrl });

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

// Listen for messages from block.js to allow proceeding
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'ALLOW_URL') {
        allowedUrls.add(message.url);
        // Automatically remove the allowance after 5 minutes for security
        setTimeout(() => {
            allowedUrls.delete(message.url);
        }, 5 * 60 * 1000);
        sendResponse({ success: true });
    }
});
