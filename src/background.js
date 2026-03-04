// Background Service Worker

const API_URL = 'http://localhost:5000/predict';

// Set of temporarily allowed URLs
const allowedUrls = new Set();

// Track the last known SAFE URL per tab so Go Back works correctly
const lastSafeUrl = new Map();

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
            // DANGEROUS — redirect to block page, pass last safe URL for Go Back
            chrome.action.setBadgeText({ text: '!', tabId });
            chrome.action.setBadgeBackgroundColor({ color: '#FF0000', tabId });

            const safeUrl = lastSafeUrl.get(tabId) || '';
            const blockUrl = chrome.runtime.getURL('block.html') +
                `?url=${encodeURIComponent(url)}` +
                `&reason=${encodeURIComponent(data.reason)}` +
                `&confidence=${encodeURIComponent(data.confidence)}` +
                `&goBack=${encodeURIComponent(safeUrl)}`;

            chrome.tabs.update(tabId, { url: blockUrl });

        } else {
            // SAFE — save this URL so Go Back can return here
            lastSafeUrl.set(tabId, url);
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

// Listen for messages from block.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'ALLOW_URL') {
        allowedUrls.add(message.url);
        setTimeout(() => { allowedUrls.delete(message.url); }, 5 * 60 * 1000);
        sendResponse({ success: true });
    }

    if (message.action === 'GO_BACK') {
        const tabId = sender.tab?.id;
        if (!tabId) return;
        // Navigate the tab directly — bypasses history to avoid re-triggering block
        const goBackTarget = message.goBackUrl || 'chrome://newtab';
        chrome.tabs.update(tabId, { url: goBackTarget });
        sendResponse({ success: true });
    }
});

// Clean up lastSafeUrl when a tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
    lastSafeUrl.delete(tabId);
});
