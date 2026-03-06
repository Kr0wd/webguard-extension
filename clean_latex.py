import re

def clean_latex(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find tabular blocks and remove empty lines inside them
    def clean_tabular(match):
        tabular_content = match.group(0)
        # Remove empty lines or lines with just spaces
        cleaned = re.sub(r'\n\s*\n', '\n', tabular_content)
        return cleaned

    # Apply to both tabular and tabularx if any exist
    new_content = re.sub(r'\\begin{tabular}.*?\\end{tabular}', clean_tabular, content, flags=re.DOTALL)
    
    # Also remove any remaining \[H\] and change them to \[htbp\]
    new_content = new_content.replace('[H]', '[htbp]')
    
    # Also ensure there are no stray empty lines before \end{frame}
    # (not strictly required but good practice)
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print("Cleaned up tabular environments and float placements in sample.tex")
    
if __name__ == '__main__':
    clean_latex('/home/krowd/webguard-extension/sample.tex')
