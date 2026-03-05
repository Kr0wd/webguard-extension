import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_uml_diagram():
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # UML Box helper
    def draw_class(x, y, w, h, title, attributes, methods, bg_color='#fffbe6'):
        # Main box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=bg_color, zorder=2)
        ax.add_patch(rect)
        
        # Title box
        title_h = h * 0.18
        rect_title = patches.Rectangle((x, y + h - title_h), w, title_h, linewidth=2, edgecolor='black', facecolor='#d9edf7', zorder=3)
        ax.add_patch(rect_title)
        
        # Title text
        ax.text(x + w/2, y + h - title_h/2, title, ha='center', va='center', weight='bold', fontsize=11, zorder=4)
        
        # Divider for attributes / methods
        mid_y = y + (h - title_h) * 0.45
        ax.plot([x, x+w], [mid_y, mid_y], color='black', lw=1.5, zorder=3)
        
        # Attributes
        ax.text(x + 2, y + h - title_h - 1, "\n".join(attributes), ha='left', va='top', fontsize=9, family='monospace', zorder=4)
        
        # Methods
        ax.text(x + 2, mid_y - 1, "\n".join(methods), ha='left', va='top', fontsize=9, family='monospace', zorder=4)
        
        # Return connection points: Top, Bottom, Left, Right
        return (x + w/2, y + h), (x + w/2, y), (x, y + h/2), (x + w, y + h/2)

    def draw_line(p1, p2, label="", style="-", arrowhead="-|>"):
        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle=arrowhead, ls=style, color='black', lw=1.5, shrinkA=0, shrinkB=0), zorder=1)
        if label:
            ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, label, ha='center', va='center', backgroundcolor='white', fontsize=9, zorder=3)

    plt.title("UML Class Diagram: WebGuard System Components", fontsize=18, weight='bold', pad=20)

    # ==========================================
    # DRAW CLASSES
    # ==========================================
    
    # 1. Chrome Extension (Frontend)
    top1, bot1, left1, right1 = draw_class(5, 72, 22, 22, "ChromeExtension UI", 
               ["- activeUrl: String", "- isWarningShown: boolean", "- currentStats: Object"], 
               ["+ onUrlIntercept()", "+ renderPopup()", "+ showWarningScreen()", "+ allowBypass()"])

    # 2. Background Service
    top2, bot2, left2, right2 = draw_class(38, 72, 24, 22, "BackgroundWorker (Service)", 
               ["- API_ENDPOINT: String", "- localCache: Map"], 
               ["+ listener_onBeforeRequest()", "+ fetchServerPrediction(url)", "+ enforceBlockingRules()"])

    # 3. Flask Server API
    top3, bot3, left3, right3 = draw_class(38, 35, 24, 24, "FlaskServerAPI (Controller)", 
               ["- BASE_DIR: String", "- model_loaded: boolean", "- MAX_LEN: int = 550"], 
               ["+ start_server()", "+ route_predict(POST)", "+ route_health(GET)"])

    # 4. URL Normalizer & Filter
    top4, bot4, left4, right4 = draw_class(5, 35, 23, 24, "ShieldFilter (Layer 1)", 
               ["- WHITELIST_SET: HashSet", "- SAFE_TLDS: Tuple"], 
               ["+ strip_protocol(url) : str", "+ is_whitelisted(url) : bool", "+ is_chrome_internal() : bool"])

    # 5. Static Heuristics
    top5, bot5, left5, right5 = draw_class(70, 60, 27, 24, "StaticHeuristics (Layer 2)", 
               ["- SUSPICIOUS_REGEX: List<Pattern>", "- SAFE_CDN_DOMAINS: Set"], 
               ["+ check_xss_injection(url)", "+ check_sql_injection(url)", "+ check_path_traversal(url)", "+ check_static_assets()"])

    # 6. Feature Extractor
    top6, bot6, left6, right6 = draw_class(70, 30, 27, 24, "FeatureExtractor", 
               ["- TARGET_BRANDS: List<String>", "- HIGH_TRUST: HashSet<String>"], 
               ["+ calculate_entropy(str) : float", "+ count_symbols(url) : array", "+ check_fuzzy_brand_spoof()", "+ build_13_feature_vector()"])

    # 7. AI Ensemble
    top7, bot7, left7, right7 = draw_class(38, 5, 24, 22, "AIEnsemble (Layer 3)", 
               ["- svm_model: LinearSVC", "- cnn_model: Sequential", "- xgb_meta_model: XGBoost"], 
               ["+ load_models_from_disk()", "+ predict_svm(tfidf_vec)", "+ predict_cnn(padded_seq)", "+ predict_meta(combined_features)"])

    # ==========================================
    # DRAW RELATIONS
    # ==========================================
    
    # Extension UI to Worker (Communication)
    draw_line(right1, left2, "Message Passing", arrowhead="-", style="solid")
    
    # Worker to API (REST HTTP)
    draw_line(bot2, top3, "HTTP POST JSON", arrowhead="-|>", style="dashed")
    
    # API to Filter (Dependency)
    draw_line(left3, right4, "Dependency", arrowhead="-|>", style="dashed")
    
    # API to Heuristics (Dependency)
    draw_line((right3[0], right3[1]+8), left5, "Dependency", arrowhead="-|>", style="dashed")
    
    # API to AI Feat Extractor (Dependency)
    draw_line((right3[0], right3[1]-8), left6, "Dependency", arrowhead="-|>", style="dashed")
    
    # API to AI Ensemble (Uses)
    draw_line(bot3, top7, "Uses", arrowhead="-|>", style="solid")

    # Feature Extractor to AI Ensemble (Data passing)
    draw_line(bot6, right7, "NumPy Array", arrowhead="->", style="dotted")

    plt.tight_layout()
    plt.savefig('docs/uml_diagram.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_uml_diagram()
    print("UML diagram successfully saved to docs/uml_diagram.png")
