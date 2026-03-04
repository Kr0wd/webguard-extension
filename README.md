# WebGuard: Advanced Threat Detection Browser Extension

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![React](https://img.shields.io/badge/react-18-blue)]()

WebGuard is a state-of-the-art, machine-learning-powered browser extension designed to protect users from phishing, malware, and malicious URLs in real-time. By leveraging a multi-layered hybrid architecture, it detects zero-day threats, intelligent brand-spoofing attempts, and embedded web vulnerabilities before the user is compromised.

## 🚀 Key Features

*   **🛡️ Multi-Layered Threat Detection Engine**: Combines static heuristics, whitelist checking, and an advanced AI ensemble to categorize threats.
*   **🧠 Hybrid AI Ensemble Pipeline**: 
    *   **Character-level CNN-BiLSTM**: For deep sequential URL analysis.
    *   **Support Vector Machine (SVM)**: For rapid, high-precision lightweight classification.
    *   **XGBoost Meta-Learner**: Stacked generalizer that takes probabilities from the base models and handcrafted features to make the final determination.
*   **⚡ Real-Time Processing**: The Flask-based inference server executes the pipeline in milliseconds, providing an invisible yet robust shield.
*   **🎯 Zero-Day Protection**: Safe Path Heuristics and dynamic threat scoring provide high recall against novel phishing tactics while maintaining >99% precision.
*   **🎨 Seamless User Experience**: Built on a modern React + Vite frontend, injecting a beautiful, user-friendly UI component directly into your browsing experience.

## 🏗️ Architecture Overview

When a user navigates to a URL, WebGuard executes the following layered pipeline:

1.  **The Shield Structure (Layer 1)**: O(1) whitelist checking against known safe internal domains and global high-trust entities.
2.  **Static Heuristics (Layer 2)**: Regex-based pattern matching instantly flags obvious XSS, SQLi, Command Injection, and Path Traversal attempts.
3.  **Global Meta-Learner (Layer 3)**:
    *   Extracts 13 statistical and structural features (e.g., entropy, brand spoof indicators, digit ratios).
    *   Passes standardized features into the SVM and sequential sequences into the CNN model.
    *   The XGBoost Meta-Model processes the aggregated predictions to output the final malicious probability.

## 📁 Repository Structure

```text
webguard-extension/
├── models/             # Contains the pre-trained SVM, CNN, and Meta-Learner models (.pkl, .keras)
├── data/               # Contains datasets for training and validation
├── src/                # React Frontend source code for the extension
├── public/             # Static assets for the extension
├── server.py           # Production-ready Flask inference server
├── train.py            # Model training and pipeline generation script
├── evaluate.py         # Testing script for measuring model metrics (F1, Precision, Recall)
├── INSTALLATION.md     # Detailed setup and installation instructions
└── manifest.json       # Browser Extension Manifest V3 configuration
```

## 🛠️ Installation & Setup

For detailed instructions on setting up both the Python inference server and compiling the React browser extension, please refer to the [Installation Guide](INSTALLATION.md).

## 📊 Model Performance

Our comprehensive evaluation on a balanced dataset of over 100,000 URLs yields the following validated metrics (using `evaluate.py`):

*   **Accuracy**: > 99.0%
*   **Precision**: > 98.7%
*   **Recall**: > 99.2%
*   **F1-Score**: > 98.9%

## 🤝 Contributing

We welcome contributions from researchers and developers! Please follow these guidelines:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
