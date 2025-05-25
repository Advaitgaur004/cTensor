import os
import re
import sys
from datetime import datetime
import glob

def extract_accuracy(content):
    # Look for accuracy pattern like "accuracy: 0.1234"
    accuracy_match = re.search(r'accuracy:\s*([\d.]+)', content)
    if accuracy_match:
        return float(accuracy_match.group(1))
    return None

def extract_loss_values(content):
    # Look for patterns like "Epoch X average loss: Y.ZZZZZZ"
    loss_matches = re.findall(r'Epoch\s+\d+\s+average\s+loss:\s*([\d.]+)', content)
    return [float(loss) for loss in loss_matches]

def extract_platform_info(content):
    # Try different patterns for platform
    platform_match = re.search(r'[Pp]latform:\s*(.+)', content)
    platform = platform_match.group(1).strip() if platform_match else "Unknown"
    
    # Try different patterns for compiler  
    compiler_match = re.search(r'[Cc]ompiler:\s*(.+)', content)
    compiler = compiler_match.group(1).strip() if compiler_match else "Unknown"
    
    # Try different patterns for framework
    framework_match = re.search(r'[Ff]ramework:\s*(.+)', content)
    if framework_match:
        framework = framework_match.group(1).strip()
    elif "pytorch results" in content.lower():
        framework = "PyTorch"
    else:
        framework = "cTensor"
    
    return platform, compiler, framework

def analyze_results():
    print("=== comparing my lib with pytorch ===")
    print(f"time: {datetime.now()}")
    print("")
    
    # Find all result files
    result_files = glob.glob("results.txt") + glob.glob("pytorch_results.txt") + glob.glob("*results*.txt")
    # Remove duplicates
    result_files = list(set(result_files))
    
    if not result_files:
        print("ERROR: No result files found!")
        print("Available files:", os.listdir("."))
        return False
    
    results = []
    
    # Process each result file
    for file_path in result_files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            platform, compiler, framework = extract_platform_info(content)
            accuracy = extract_accuracy(content)
            loss_values = extract_loss_values(content)
            
            results.append({
                'file': file_path,
                'platform': platform,
                'compiler': compiler,
                'framework': framework,
                'accuracy': accuracy,
                'loss_values': loss_values,
                'content': content
            })
            
            print(f"  Platform: {platform}")
            print(f"  Compiler: {compiler}")
            print(f"  Framework: {framework}")
            print(f"  Accuracy: {accuracy}")
            print(f"  Loss values: {loss_values}")
            print("")
            
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            print("")
    
    # simple comparison
    print("=== results ===")
    print("")
    
    # find pytorch vs lib
    my_results = [r for r in results if r['framework'] != 'PyTorch']
    pytorch_results = [r for r in results if r['framework'] == 'PyTorch']
    
    if pytorch_results:
        pytorch_acc = pytorch_results[0]['accuracy']
        print(f"pytorch accuracy: {pytorch_acc}")
    
    print("my lib results:")
    for result in my_results:
        acc_diff = ""
        if pytorch_results and result['accuracy'] is not None and pytorch_results[0]['accuracy'] is not None:
            diff = abs(result['accuracy'] - pytorch_results[0]['accuracy'])
            acc_diff = f" (diff: {diff:.3f})"
        
        print(f"  {result['platform']} {result['compiler']}: {result['accuracy']}{acc_diff}")
    
    # quick check
    accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]
    if len(accuracies) > 1:
        spread = max(accuracies) - min(accuracies)
        if spread < 0.05:
            print("looks consistent :)")
        else:
            print("hmm results vary a lot :(")
    
    print("")
    
    # save everything to file
    with open("comparison_report.txt", "w") as f:
        f.write("comparison results\n")
        f.write(f"generated: {datetime.now()}\n\n")
        
        for result in results:
            f.write(f"=== {result['file']} ===\n")
            f.write(f"platform: {result['platform']}\n")
            f.write(f"compiler: {result['compiler']}\n")
            f.write(f"accuracy: {result['accuracy']}\n")
            f.write("\nfull output:\n")
            f.write(result['content'])
            f.write("\n" + "-"*30 + "\n\n")
    
    print("saved detailed report")
    return True

if __name__ == "__main__":
    success = analyze_results()
    sys.exit(0 if success else 1)