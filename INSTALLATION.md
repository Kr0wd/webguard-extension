# URL Safety Checker Browser Extension

A browser extension that uses machine learning to predict whether URLs are dangerous or safe.

## Features

- 🛡️ Real-time URL safety checking
- 🤖 ML-powered predictions using SVM model
- 🎨 Beautiful, modern UI
- ⚡ Fast and lightweight

## Prerequisites

- Python 3.7+
- Node.js 16+
- Chrome or Firefox browser

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd extension
pip install -r requirements.txt
```

### 2. Start the Flask API Server

The extension requires a local Flask server to run the ML model:

```bash
python server.py
```

The server will start on `http://localhost:5000`. Keep this terminal window open.

### 3. Build the Extension

In a new terminal:

```bash
cd extension
npm install
npm run build
```

This will create a `dist` folder with the extension files.

### 4. Copy manifest.json to dist

After building, copy the manifest file:

```bash
cp manifest.json dist/
```

### 5. Load the Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension/dist` folder
5. The extension icon should appear in your toolbar

### 6. Load the Extension in Firefox

1. Open Firefox and go to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Navigate to `extension/dist` and select the `manifest.json` file

## Usage

1. **Make sure the Flask server is running** (`python server.py`)
2. Navigate to any website
3. Click the extension icon in your browser toolbar
4. The extension will automatically analyze the current URL and display:
   - Whether the URL is safe or dangerous
   - Confidence score (if available)

## Model Information

The extension uses a character-based SVM model (`char_svm_model.pkl`) located in the `public/` folder. The model analyzes various features of the URL including:

- URL length
- Special character counts
- Digit and letter ratios
- Domain characteristics

**Important Note**: The feature extraction in `server.py` is a basic implementation. You may need to adjust the `extract_features()` function to match how your model was trained for optimal accuracy.

## Development

To run in development mode:

```bash
# Terminal 1: Start Flask server
python server.py

# Terminal 2: Start Vite dev server
npm run dev
```

Then open `http://localhost:5173` in your browser to test the UI.

## Troubleshooting

### "Error: Failed to get prediction"

- Make sure the Flask server is running on port 5000
- Check that `char_svm_model.pkl` exists in the `public/` folder
- Verify Python dependencies are installed

### Extension doesn't load

- Make sure you've built the extension (`npm run build`)
- Ensure `manifest.json` is copied to the `dist/` folder
- Check browser console for errors

### Model predictions seem incorrect

- The feature extraction may need adjustment based on your model's training
- Review and modify the `extract_features()` function in `server.py`

## Project Structure

```
extension/
├── manifest.json          # Extension manifest (Manifest V3)
├── server.py             # Flask API server for model predictions
├── requirements.txt      # Python dependencies
├── package.json          # Node.js dependencies
├── vite.config.js        # Vite build configuration
├── public/
│   └── char_svm_model.pkl  # ML model file
├── src/
│   ├── App.jsx           # Main React component
│   ├── App.css           # Styles
│   └── main.jsx          # Entry point
└── dist/                 # Built extension (after npm run build)
```

## Security Note

This extension makes requests to `localhost:5000`. In a production environment, you would want to:
- Host the model on a secure server
- Use HTTPS
- Implement proper authentication
- Add rate limiting

## License

MIT
