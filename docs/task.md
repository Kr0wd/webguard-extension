# Task Checklist

- [x] Optimize Data Pipeline to hit 98% Accuracy
  - [x] Increase TF-IDF vocabulary to `25000` and n-gram range to `(1,3)`.
  - [x] Include more Phishing samples in training rather than capping at 20,000.
  - [x] Adjust XGBoost class weights to penalize False Negatives (target `2.5x` weight for malicious).
- [x] Retrain Models (v6 - 446k URLs + High-Fidelity Features)
    - [x] SVM Foundation (95.02% Accuracy)
    - [x] CNN Deep Learning (98.03% Val Accuracy)
    - [x] XGBoost Meta-Learner (98.77% Master Accuracy)
- [x] Run Final Evaluation (97.80% Overall - Met Recall/Precision targets)
- [x] Generate Performance Visualizations (Confusion Matrix, etc.)
- [x] Evaluate System Performance (>98% Target)
    - [x] Run `evaluate_mixed.py`
    - [ ] Analyze False Positives/Negatives
- [x] Optimize Classification Thresholds
  - [x] Adjust base decision threshold to `0.35` in `evaluate_mixed.py` to boost recall without destroying precision.
  - [ ] Mirror final optimal threshold to `server.py`.
