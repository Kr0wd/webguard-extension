import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_dataflow_diagram():
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # --- STYLE CONSTANTS ---
    c_entity = '#2b5b84'    # Dark Blue for External Entities
    c_process = '#2e8b57'   # Sea Green for Processes
    c_store = '#b22222'     # Firebrick Red for Data Stores
    c_text = 'white'
    c_arrow = '#444444'

    # --- DRAW HELPERS ---
    def draw_entity(x, y, w, h, text):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=c_entity, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=c_text, fontsize=10, weight='bold', wrap=True)
        return (x + w/2, y + h)

    def draw_process(x, y, r, text, num):
        circle = patches.Circle((x, y), r, linewidth=2, edgecolor='black', facecolor=c_process, zorder=2)
        ax.add_patch(circle)
        # Process number
        ax.text(x, y + r - 1.5, num, ha='center', va='center', color=c_text, fontsize=9, weight='bold')
        # Process text
        ax.text(x, y, text, ha='center', va='center', color=c_text, fontsize=9, weight='bold', wrap=True)
        return (x, y)

    def draw_store(x, y, w, h, text, num):
        # Data store (open-ended rectangle with a horizontal line)
        ax.plot([x, x+w], [y+h, y+h], color='black', lw=2, zorder=2) # Top
        ax.plot([x, x+w], [y, y], color='black', lw=2, zorder=2)     # Bottom
        ax.plot([x, x], [y, y+h], color='black', lw=2, zorder=2)     # Left
        ax.plot([x+w*0.2, x+w*0.2], [y, y+h], color='black', lw=2, zorder=2) # Divider
        
        rect = patches.Rectangle((x, y), w, h, facecolor=c_store, alpha=0.9, zorder=1)
        ax.add_patch(rect)
        
        ax.text(x + w*0.1, y + h/2, num, ha='center', va='center', color=c_text, fontsize=9, weight='bold')
        ax.text(x + w*0.6, y + h/2, text, ha='center', va='center', color=c_text, fontsize=9, weight='bold')
        return (x + w/2, y + h/2)

    def draw_path_arrow(points, label=None):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        ax.plot(x_coords, y_coords, color=c_arrow, lw=2, zorder=1)
        ax.annotate('', xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="->", color=c_arrow, lw=2, shrinkA=0, shrinkB=8), zorder=1)
        
        if label:
            max_len = 0
            best_segment = None
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i+1]
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                if dist > max_len:
                    max_len = dist
                    best_segment = (p1, p2)
            
            p1, p2 = best_segment
            x_mid = (p1[0] + p2[0]) / 2
            y_mid = (p1[1] + p2[1]) / 2
            
            if p1[1] == p2[1]: # horizontal line
                y_mid += 1.8
            else: # vertical line
                x_mid += 2.8
            
            ax.text(x_mid, y_mid, label, ha='center', va='center', fontsize=8.5, backgroundcolor='#f8f9fa', color='black', weight='bold', zorder=3, bbox=dict(facecolor='#f8f9fa', edgecolor='none', pad=1))

    # ==========================================
    # ENTITIES & PROCESSES
    # ==========================================
    draw_entity(2, 45, 12, 10, "External\nEntity\nUser / Browser")

    p1 = draw_process(30, 50, 6, "Intercept &\nClean URL", "1.0")
    p2 = draw_process(50, 50, 6, "Evaluate URL\nAgainst Shield", "2.0")
    p3 = draw_process(70, 50, 6, "Apply Static\nHeuristics", "3.0")
    
    p4 = draw_process(70, 20, 6, "Extract Num\nFeatures", "4.0")
    p5 = draw_process(90, 50, 6, "Predict with\nBase Models", "5.0")
    p6 = draw_process(90, 20, 6, "Meta-Ensemble\nPrediction", "6.0")

    # ==========================================
    # DATA STORES 
    # ==========================================
    draw_store(42, 75, 16, 6, "Trusted List\n(O(1) Set)", "D1")
    draw_store(62, 75, 16, 6, "Regex Attack\nSignatures", "D2")
    
    draw_store(82, 75, 16, 6, "Trained\nCNN & SVM", "D3")
    draw_store(82, 5, 16, 6, "Trained\nXGBoost", "D4")

    # ==========================================
    # DATA FLOWS (Perfectly orthogonal bounds)
    # ==========================================
    # P1 <-> User 
    draw_path_arrow([(14, 52), (24, 52)], "Raw URL ")
    draw_path_arrow([(24, 48), (14, 48)], "Allow Local")

    # P1 -> P2 
    draw_path_arrow([(36, 50), (44, 50)], "Decoded URL")
    
    # D1 <-> P2 
    draw_path_arrow([(48, 56), (48, 75)], "Lookup")
    draw_path_arrow([(52, 75), (52, 56)], "Trusted Domains")
    
    # P2 -> User
    draw_path_arrow([(50, 44), (50, 40), (10, 40), (10, 45)], "Safe (Whitelisted URL)")

    # P2 -> P3 
    draw_path_arrow([(56, 50), (64, 50)], "Unknown URL")

    # D2 -> P3 
    draw_path_arrow([(70, 75), (70, 56)], "Static Rules")

    # P3 -> User 
    draw_path_arrow([(70, 44), (70, 35), (6, 35), (6, 45)], "Dangerous / Safe (CDN)")

    # P3 -> P4 
    draw_path_arrow([(76, 50), (80, 50), (80, 30), (70, 30), (70, 26)], "Complex URL")

    # P4 -> P5 
    draw_path_arrow([(68, 26), (68, 30), (60, 30), (60, 64), (88, 64), (88, 56)], "Clean URL") 
    
    # P4 -> P6 
    draw_path_arrow([(76, 20), (84, 20)], "13 Mathematical Features")

    # D3 -> P5 
    draw_path_arrow([(92, 75), (92, 56)], "Model Weights")
    
    # P5 -> P6 
    draw_path_arrow([(90, 44), (90, 26)], "SVM/CNN Probs")
    
    # D4 -> P6 
    draw_path_arrow([(90, 11), (90, 14)], "XGB Tree")

    # P6 -> User 
    draw_path_arrow([(96, 20), (102, 20), (102, 2), (12, 2), (12, 45)], "Final JSON Prediction {'is_dangerous': T/F}")

    # Add Legend
    legend_x = 75
    legend_y = 80
    ax.add_patch(patches.Rectangle((legend_x, legend_y), 32, 18, facecolor='white', edgecolor='black', zorder=5))
    ax.text(legend_x + 16, legend_y + 15, "Legend", ha='center', weight='bold', fontsize=10, zorder=6)
    
    ax.add_patch(patches.Rectangle((legend_x + 2, legend_y + 9), 6, 4, facecolor=c_entity, edgecolor='black', zorder=6))
    ax.text(legend_x + 10, legend_y + 11, "External Entity (Source/Sink)", va='center', fontsize=9, zorder=6)
    
    ax.add_patch(patches.Circle((legend_x + 5, legend_y + 6), 2, facecolor=c_process, edgecolor='black', zorder=6))
    ax.text(legend_x + 10, legend_y + 6, "Process (System Action)", va='center', fontsize=9, zorder=6)
    
    ax.add_patch(patches.Rectangle((legend_x + 2, legend_y + 1), 6, 3, facecolor=c_store, edgecolor='black', zorder=6))
    ax.text(legend_x + 10, legend_y + 2.5, "Data Store (Database/File)", va='center', fontsize=9, zorder=6)

    plt.tight_layout()
    plt.savefig('docs/dataflow_diagram.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_dataflow_diagram()
    print("DFD successfully saved to docs/dataflow_diagram.png")
