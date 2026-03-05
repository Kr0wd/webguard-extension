import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Predefined colors
    c_dataset = '#e8f4f8'
    c_prep = '#fff2e6'
    c_model = '#e6ffed'
    c_ensemble = '#f0e6ff'
    c_output = '#ffe6e6'
    edge_color = '#333333'
    
    # Helper to draw boxes
    def draw_box(x, y, w, h, text, color, fontsize=10, weight='normal'):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge_color, facecolor=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, weight=weight, wrap=True, zorder=3)
        return (x + w, y + h/2), (x + w/2, y) # Return right_mid, bottom_mid

    def draw_arrow(start, end, label=None):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color=edge_color, lw=2, shrinkA=5, shrinkB=5), zorder=1)
        if label:
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2, label, 
                    ha='center', va='center', fontsize=8, backgroundcolor='white', zorder=4)

    # 1. Datasets Section (Left)
    ax.text(12, 95, "1. THE 'TITAN-98' DATASET POOL (446k+ URLs)", fontsize=14, weight='bold', ha='center')
    
    ds_w, ds_h = 20, 8
    ds_x = 2
    y_starts = [85, 75, 65, 55, 45, 35]
    ds_labels = [
        "Definitive Benign\n(Majestic + Cisco)\n[120k Samples]",
        "Definitive Malicious\n(Active Phish DB)\n[80k Samples]",
        "PhishTank Archive\n(Typosquats)",
        "CAPEC Dataset\n(Injection/XSS/SQLi)",
        "New Unseen Data\n(Zero-Day Bridges)",
        "Merged Traffic\n(General Noise)"
    ]
    
    ds_outputs = []
    for y, label in zip(y_starts, ds_labels):
        right_mid, _ = draw_box(ds_x, y, ds_w, ds_h, label, c_dataset, 9)
        ds_outputs.append((right_mid[0], right_mid[1]))

    # 2. Data Balancing Section (Middle-Left)
    bal_x = ds_w + ds_x + 8
    bal_y = 52
    bal_w, bal_h = 16, 20
    draw_box(bal_x, bal_y, bal_w, bal_h, "Layer 0: Data Preparation\n\n- Stratified Sampling\n- Class Balancing\n- URL Normalization\n\n(Balanced Training Set)", c_prep, 10, 'bold')
    
    for ds_out in ds_outputs:
        target_y = bal_y + bal_h/2 if bal_y < ds_out[1] < bal_y+bal_h else (bal_y+bal_h if ds_out[1] >= bal_y+bal_h else bal_y)
        draw_arrow(ds_out, (bal_x, target_y))

    # 3. Model Training Section (Middle-Right)
    ax.text(65, 95, "2. THE HYBRID META-ENSEMBLE", fontsize=14, weight='bold', ha='center')
    
    mod_x = bal_x + bal_w + 10
    
    # SVM
    svm_y = 75
    svm_w, svm_h = 22, 10
    rm_svm, bm_svm = draw_box(mod_x, svm_y, svm_w, svm_h, "Brain 1: SVM\n(Support Vector Machine)\n\nTF-IDF Vectorization\n(Keyword Specialist)", c_model, 10)
    draw_arrow((bal_x + bal_w, bal_y + bal_h*0.75), (mod_x, svm_y + svm_h/2), "Text Data")
    
    # CNN
    cnn_y = 55
    cnn_w, cnn_h = 22, 12
    rm_cnn, bm_cnn = draw_box(mod_x, cnn_y, cnn_w, cnn_h, "Brain 2: CNN + BiLSTM\n(Deep Neural Network)\n\nCharacter Embeddings\nMaxLen = 550\n(Context & Sequence Reader)", c_model, 10)
    draw_arrow((bal_x + bal_w, bal_y + bal_h/2), (mod_x, cnn_y + cnn_h/2), "Char Sequences")
    
    # Features
    feat_y = 35
    feat_w, feat_h = 22, 10
    rm_feat, bm_feat = draw_box(mod_x, feat_y, feat_w, feat_h, "Brain 3: Feature Extraction\n\n13 Custom Math Features\n(Entropy, Brand Spoofing,\nURL Structure Metrics)", c_model, 10)
    draw_arrow((bal_x + bal_w, bal_y + bal_h*0.25), (mod_x, feat_y + feat_h/2), "Raw URLs")

    # 4. Meta Learner (Right)
    meta_x = mod_x + svm_w + 12
    meta_y = 48
    meta_w, meta_h = 16, 26
    draw_box(meta_x, meta_y, meta_w, meta_h, "The Judge: XGBoost\n(Meta-Learner)\n\nTakes predictions from\nSVM & CNN + Features.\n\nTrained w/ 2.5x\nRecall Penalty\nfor Malicious URLs", c_ensemble, 10, 'bold')
    
    # Arrows to Meta
    draw_arrow((mod_x + svm_w, svm_y + svm_h/2), (meta_x, meta_y + meta_h*0.8), "Probabilities")
    draw_arrow((mod_x + cnn_w, cnn_y + cnn_h/2), (meta_x, meta_y + meta_h*0.5), "Probabilities")
    draw_arrow((mod_x + feat_w, feat_y + feat_h/2), (meta_x, meta_y + meta_h*0.2), "Scaled Features")
    
    # 5. Output
    out_x = meta_x + 2
    out_y = 15
    out_w, out_h = 12, 10
    draw_box(out_x, out_y, out_w, out_h, "Final Verdict\n(0.0 to 1.0)\n\nThreat Detection\nOutput", c_output, 10, 'bold')
    
    draw_arrow((meta_x + meta_w/2, meta_y), (out_x + out_w/2, out_y + out_h), "Threshold Eval")

    plt.tight_layout()
    
    # Save image
    output_path = "/home/krowd/webguard-extension/docs/webguard_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

create_architecture_diagram()
