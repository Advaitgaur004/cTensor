import os
import re
import glob
from datetime import datetime

def extract_accuracy(content):
    """Extract accuracy value from result content"""
    accuracy_match = re.search(r'accuracy:\s*([\d.]+)', content)
    if accuracy_match:
        return float(accuracy_match.group(1))
    return None

def extract_platform_compiler(content):
    """Extract platform and compiler info"""
    platform_match = re.search(r'platform:\s*(\w+)', content)
    compiler_match = re.search(r'compiler:\s*(\w+)', content)
    
    platform = platform_match.group(1) if platform_match else "unknown"
    compiler = compiler_match.group(1) if compiler_match else "unknown"
    
    return platform, compiler

def is_pytorch_result(content):
    """Check if this is a PyTorch result file"""
    return "pytorch results" in content or "framework: pytorch" in content

def analyze_results():
    print("=== CTensor vs PyTorch Comparison ===")
    print(f"Time: {datetime.now()}")
    print()
    
    # Find all result files
    result_files = glob.glob("*.txt")
    result_files = [f for f in result_files if "results" in f.lower()]
    
    if not result_files:
        print("ERROR: No result files found!")
        return False
    
    pytorch_accuracy = None
    ctensor_results = []
    
    # Process each file
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            accuracy = extract_accuracy(content)
            if accuracy is None:
                continue
                
            if is_pytorch_result(content):
                pytorch_accuracy = accuracy
                print(f"PyTorch baseline accuracy: {accuracy:.4f}")
            else:
                platform, compiler = extract_platform_compiler(content)
                ctensor_results.append({
                    'platform': platform,
                    'compiler': compiler,
                    'accuracy': accuracy,
                    'file': file_path
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print()
    
    if pytorch_accuracy is None:
        print("WARNING: No PyTorch baseline found!")
        return False
    
    if not ctensor_results:
        print("WARNING: No CTensor results found!")
        return False
    
    # Compare results
    print("CTensor Results vs PyTorch:")
    print("-" * 50)
    
    all_consistent = True
    tolerance = 0.01  # 1% tolerance
    
    for result in ctensor_results:
        diff = abs(result['accuracy'] - pytorch_accuracy)
        diff_pct = (diff / pytorch_accuracy) * 100
        
        status = "✓ PASS" if diff < tolerance else "✗ FAIL"
        
        print(f"{result['platform']:>8} ({result['compiler']:>5}): {result['accuracy']:.4f} "
              f"(diff: {diff:+.4f}, {diff_pct:+.2f}%) {status}")
        
        if diff >= tolerance:
            all_consistent = False
    
    print("-" * 50)
    
    if all_consistent:
        print("All platforms are consistent with PyTorch!")
    else:
        print("Some platforms show significant differences from PyTorch")
    
    # Calculate overall statistics
    accuracies = [r['accuracy'] for r in ctensor_results]
    if len(accuracies) > 1:
        spread = max(accuracies) - min(accuracies)
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"\nCTensor Statistics:")
        print(f"  Average accuracy: {avg_acc:.4f}")
        print(f"  Accuracy spread: {spread:.4f}")
        print(f"  Platforms tested: {len(ctensor_results)}")
    
    return True

if __name__ == "__main__":
    success = analyze_results()
    exit(0 if success else 1)