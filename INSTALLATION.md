# WebGuard Installation Guide

This guide walks you through setting up the **WebGuard Phishing Detection Browser Extension**. Setup is broken down into two components: the **Python Inference Server** and the **React + Vite Browser Extension**.

## Prerequisites

*   **Python:** 3.8 or higher
*   **Node.js:** v16.0 or higher
*   **NPM:** v7.0 or higher
*   **Browser:** Google Chrome, Microsoft Edge, or Mozilla Firefox

---

## Part 1: Starting the Python Inference Server

The brain of WebGuard relies on a local Python Flask server that hosts the machine learning pipeline (SVM, CNN-BiLSTM, XGBoost) and handles URL classification requests.

### 1. Install Dependencies

Navigate to the project root directory and install the required Python packages. We recommend using a virtual environment:

```bash
# Optional but recommended: Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required dependencies
pip install Flask flask-cors pandas joblib numpy tensorflow scikit-learn xgboost
```
*(Alternatively, you can run `pip install -r requirements.txt` if a requirements file is present)*

### 2. Verify Models are Present

Ensure that the required pre-trained model files exist in the `models/` directory:
*   `local_hybrid_model.keras`
*   `local_svm_model.pkl`
*   `local_meta_learner_global.pkl`
*   `local_tokenizer.pkl`, `local_label_encoder.pkl`, `local_vectorizer.pkl`, `local_url_scaler.pkl`

*(If these files are missing, you will need to generate them by running `python train.py`)*

### 3. Run the Server

Start the Flask application:

```bash
python server.py
```

The server will initialize the Hybrid Engine, load the models, and start listening on `http://localhost:5000`. Leave this terminal running in the background.

---

## Part 2: Building the Browser Extension

WebGuard's frontend is a React application built with Vite, injecting an interface into your browser to communicate with the Python server.

### 1. Install Node Packages

Open a **new terminal window** in the project root directory and install the necessary Node.js dependencies:

```bash
npm install
```

### 2. Build the Extension File

Compile the React project into static files suitable for a browser extension:

```bash
npm run build
```

This command generates a `dist/` directory containing the optimized application and automatically includes the `manifest.json`.

---

## Part 3: Loading the Extension into Your Browser

### For Google Chrome & Microsoft Edge (Chromium)

1. Open your browser and navigate to the Extensions page:
   * **Chrome:** `chrome://extensions/`
   * **Edge:** `edge://extensions/`
2. Toggle on **Developer mode** (typically located in the top-right corner).
3. Click the **Load unpacked** button.
4. Select the `dist/` folder located inside your `webguard-extension` directory.
5. Make sure the WebGuard extension is toggled **On**. You should now see the WebGuard shield icon in your extension toolbar.

### For Mozilla Firefox

1. Open Firefox and navigate to `about:debugging#/runtime/this-firefox`
2. Click **Load Temporary Add-on...**
3. Navigate to the `dist/` folder in your project directory and select the `manifest.json` file.
4. The extension is now active for your current browsing session.

---

## Troubleshooting

*   **Prediction Errors / Fails to Load:** Ensure your Python Flask server (`server.py`) is running and accessible at `http://localhost:5000`. You can test this by navigating to `http://localhost:5000/health` in your browser.
*   **"Models not loaded" Error:** Check the terminal running `server.py` for specific tracebacks. You might be missing standard dependencies like `xgboost` or the models simply haven't been generated in the `models/` folder.
*   **Changes not reflecting in UI:** If you made changes to the React code in `src/`, make sure to run `npm run build` again, go to your browser's extension panel, and click the "Refresh" icon on the extension card.

## Development Mode

If you are developing the UI component and want Hot Module Replacement (HMR), you can run:

```bash
npm run dev
```
Then navigate to `http://localhost:5173` to test the UI standalone without loading it as an extension.
