import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_comprehensive_architecture():
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # ----------------------------------------------------
    # Colors
    # ----------------------------------------------------
    c_dataset = '#e8f4f8' # Light blue
    c_train = '#dcedc1' # Light green
    c_ext = '#fff2e6' # Light orange
    c_server = '#e6e6ff' # Light purple
    c_layer1 = '#ffcccc' # Light red
    c_layer2 = '#ffe6cc' # Peach
    c_layer3 = '#cce6ff' # Sky blue
    c_model = '#d9f2d9'  # Mint
    edge_color = '#2c3e50'
    
    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def draw_box(x, y, w, h, text, color, fontsize=9, weight='normal', style='solid', fontstyle='normal'):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge_color, facecolor=color, linestyle=style, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, weight=weight, style=fontstyle, wrap=True, zorder=3)
        return (x, y, w, h)

    def draw_arrow(start, end, label=None, arc=False):
        style = "->"
        conn_style = "arc3,rad=0.2" if arc else "arc3,rad=0.0"
        
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, connectionstyle=conn_style, color=edge_color, lw=2, shrinkA=5, shrinkB=5), zorder=1)
        if label:
            x_mid, y_mid = (start[0]+end[0])/2, (start[1]+end[1])/2
            if arc:
                y_mid += 3
            ax.text(x_mid, y_mid, label, ha='center', va='center', fontsize=8, backgroundcolor='white', zorder=4, weight='bold')

    # TITLE
    ax.text(75, 98, "WEBGUARD V6: COMPREHENSIVE SYSTEM ARCHITECTURE", fontsize=20, weight='bold', ha='center')
    
    # =========================================================================================
    # PHASE 1: THE DATA ENGINE & TRAINING PIPELINE (Bottom Left)
    # =========================================================================================
    draw_box(2, 5, 45, 40, "", 'none', style='dashed')
    ax.text(24.5, 46, "PART 1: DATA ENGINE & AI TRAINING (train.py)", fontsize=12, weight='bold', ha='center')

    # Datasets
    ds_h = 4
    draw_box(4, 38, 12, ds_h, "1. Benign (120k)\n(Majestic Million)", c_dataset)
    draw_box(18, 38, 12, ds_h, "2. Malicious (80k)\n(Active Phish DB)", c_dataset)
    draw_box(32, 38, 12, ds_h, "3. PhishTank\n(Typosquat DB)", c_dataset)
    draw_box(4, 32, 12, ds_h, "4. CAPEC\n(Injection/SQLi)", c_dataset)
    draw_box(18, 32, 12, ds_h, "5. New Unseen\n(Zero-Day Bridges)", c_dataset)
    draw_box(32, 32, 12, ds_h, "6. Merged Noise\n(Real Traffic)", c_dataset)

    # Preprocessing
    draw_arrow((24.5, 32), (24.5, 29))
    draw_box(8, 24, 33, 5, "Layer 0: Stratified Sampling & Preprocessing\n(O(1) Clean, TF-IDF Vectorize, Character Padding)", c_train, 10, 'bold')

    # Models Trained
    draw_arrow((24.5, 24), (24.5, 20))
    # SVM
    draw_box(4, 12, 12, 8, "Brain 1: SVM\n\n- TF-IDF\n- Fast Keyword\ntracking", c_model)
    # CNN
    draw_box(18, 12, 13, 8, "Brain 2: CNN-BiLSTM\n\n- Char Embeddings\n- Spatial Sequencing", c_model)
    # XGBoost
    draw_box(33, 12, 13, 8, "Brain 3: XGBoost\n(Meta-Learner)\n- 2.5x Recall Penalty", c_model)
    
    draw_arrow((24.5, 12), (24.5, 9))
    draw_box(8, 6, 33, 3, "Output: Saved .pkl and .keras Models", c_train, 9, 'bold')

    
    # =========================================================================================
    # PHASE 2: CHROME EXTENSION FRONTEND (Top Left)
    # =========================================================================================
    draw_box(2, 55, 40, 35, "", 'none', style='dashed')
    ax.text(22, 91, "PART 2: USER INTERFACE (Chrome Extension)", fontsize=12, weight='bold', ha='center')

    # Browser
    draw_box(4, 78, 14, 10, "User Browser\n\nClicks a Link or\nTypes a URL", '#ffffff')
    draw_arrow((18, 83), (25, 83), "Web Request")

    # Background Script
    draw_box(25, 76, 15, 12, "background.js\n\nIntercepts URL via\nchrome.webRequest\nbefore page loads.", c_ext)
    
    # UI Components
    draw_box(4, 60, 16, 12, "React UI Components\n\n- Popup Panel\n- Whitelist Manager\n- Threat Stats", c_ext)
    draw_box(22, 60, 16, 12, "Warning Screen\n(warning.html)\n\nBlocks Rendering\nShows AI Reason", '#ffe6e6')

    # Control Flow
    draw_arrow((32.5, 76), (32.5, 72), "If Blocekd")
    
    
    # =========================================================================================
    # PHASE 3: PYTHON FLASK SERVER ENGINE (Right Side)
    # =========================================================================================
    draw_box(50, 5, 95, 85, "", 'none', style='dashed')
    ax.text(97.5, 91, "PART 3: REAL-TIME THREAT DETECTION ENGINE (server.py)", fontsize=12, weight='bold', ha='center')

    # Communication Bridge
    draw_arrow((40, 85), (55, 85), "POST /predict\n{url: '...'}", arc=True)
    draw_arrow((55, 80), (40, 80), "JSON Response\n{is_dangerous: TF}", arc=True)

    # API Endpoint
    draw_box(55, 75, 85, 12, "Flask API: /predict Endpoint\n\n1. Unquote URL\n2. Strip Protocol (http://)\n3. Parse Root Domain vs Subdomains", c_server, 10, 'bold')
    draw_arrow((97.5, 75), (97.5, 71))

    # LAYER 1
    draw_box(55, 61, 85, 10, "LAYER 1: The Shield (O(1) Whitelist)\n\nInstant bypass for pre-verified internal subnets (localhost, 192.168.x),\nand regional Top-Level Domains (.gov.in, .nic.in, .edu.in)", c_layer1, 10)
    draw_arrow((97.5, 61), (97.5, 57))
    draw_arrow((55, 66), (42, 66), "Safe Return") # Fast Return

    # LAYER 2
    draw_box(55, 41, 85, 16, "LAYER 2: Heuristic Static Rules\n\nRegex Pattern Matching pre-built for:\n- XSS (<script>, javascript:)\n- SQL Injection (UNION SELECT, sleep())\n- Directory Traversal (../../etc/passwd)\n- Brand Spoofing (paypal.com.evil.ru style subdomains)\n- Tiered Bypass (Safe CDNs, .jpg, .css files)", c_layer2, 10)
    draw_arrow((97.5, 41), (97.5, 37))
    draw_arrow((55, 49), (42, 49), "Block / Bypass Return") # Fast Return

    # LAYER 3 HEADER
    draw_box(55, 27, 85, 10, "(If URL is unknown to Layers 1 & 2, initialize AI)", c_server, 10, 'normal', 'solid', 'italic')
    
    # AI Feature Prep
    draw_box(57, 10, 20, 15, "1. Feature Extraction\n\nGenerates 13 math\nfeatures (Entropy,\nLength, Special Chars)\n+ Brand Spoof Bool", c_layer3, 9)
    # Models loaded from disk
    draw_arrow((24.5, 6), (77, 6), "Loads Pre-Trained Models on Boot", arc=False)
    
    draw_box(80, 18, 16, 8, "2. SVM Predict\n(Keywords)", c_model)
    draw_box(80, 8, 16, 8, "2. CNN Predict\n(Sequence)", c_model)
    
    # Meta
    draw_arrow((77, 17.5), (100, 17.5))
    draw_arrow((96, 22), (100, 22))
    draw_arrow((96, 12), (100, 12))
    
    draw_box(100, 8, 22, 18, "3. XGBoost Meta-Learner\n\nCombines Extracted Features\n+ SVM Prob + CNN Prob\nto output final\nMalicious Probability.", c_model, 10, 'bold')
    
    # Threshold Logic
    draw_arrow((122, 17), (126, 17))
    draw_box(126, 8, 17, 18, "4. Threshold Tiers\n\n65% for Unknown\n85% for Trusted Docs\n99% for Short Startups", c_server, 9, 'bold')

    # Final return from Layer 3
    draw_arrow((134.5, 26), (134.5, 68))
    draw_arrow((134.5, 68), (140, 68))
    draw_arrow((140, 68), (140, 77))
    draw_arrow((140, 77), (140, 85))
    draw_arrow((140, 85), (55, 85), "Final AI Decision\n", arc=False)

    plt.tight_layout()
    output_path = "/home/krowd/webguard-extension/docs/webguard_comprehensive_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

create_comprehensive_architecture()
