"""
Run gradient direction analysis for all trained models in the study.
This script iterates through all model configurations and generates direction analysis results.
"""

import os
import subprocess
import sys

# Define all model configurations
models = [
    # Regressor
    ("Regressor", "CustomL1Loss", "Adam"),
    ("Regressor", "CustomL1Loss", "AdamW"),
    ("Regressor", "HuberLoss", "Adam"),
    ("Regressor", "HuberLoss", "AdamW"),
    ("Regressor", "MSELoss", "Adam"),
    ("Regressor", "MSELoss", "AdamW"),
    
    # BasicUNet
    ("BasicUNet", "CustomL1Loss", "Adam"),
    ("BasicUNet", "CustomL1Loss", "AdamW"),
    ("BasicUNet", "HuberLoss", "Adam"),
    ("BasicUNet", "HuberLoss", "AdamW"),
    ("BasicUNet", "MSELoss", "Adam"),
    ("BasicUNet", "MSELoss", "AdamW"),
    
    # AutoEncoder
    ("AutoEncoder", "CustomL1Loss", "Adam"),
    ("AutoEncoder", "CustomL1Loss", "AdamW"),
    ("AutoEncoder", "HuberLoss", "Adam"),
    ("AutoEncoder", "HuberLoss", "AdamW"),
    ("AutoEncoder", "MSELoss", "Adam"),
    ("AutoEncoder", "MSELoss", "AdamW"),
    
    # DenseNet169
    ("DenseNet169", "CustomL1Loss", "Adam"),
    ("DenseNet169", "CustomL1Loss", "AdamW"),
    ("DenseNet169", "HuberLoss", "Adam"),
    ("DenseNet169", "HuberLoss", "AdamW"),
    ("DenseNet169", "MSELoss", "Adam"),
    ("DenseNet169", "MSELoss", "AdamW"),
    
    # EfficientNetB4
    ("EfficientNetB4", "CustomL1Loss", "Adam"),
    ("EfficientNetB4", "CustomL1Loss", "AdamW"),
    ("EfficientNetB4", "HuberLoss", "Adam"),
    ("EfficientNetB4", "HuberLoss", "AdamW"),
    ("EfficientNetB4", "MSELoss", "Adam"),
    ("EfficientNetB4", "MSELoss", "AdamW"),
]

def main():
    total = len(models)
    successful = 0
    failed = []
    skipped = []
    
    print(f"Starting gradient direction analysis for {total} models...")
    print("=" * 80)
    print("Note: Each model may take 10-20 minutes to evaluate.")
    print("=" * 80)
    
    for i, (arch, loss, opt) in enumerate(models, 1):
        model_name = f"best_{arch}_{loss}_{opt}"
        model_path = f"models/{model_name}.pth"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\n[{i}/{total}] SKIPPED: {arch} + {loss} + {opt}")
            print(f"  Reason: Model file not found: {model_path}")
            skipped.append((arch, loss, opt))
            continue
        
        # Check if analysis already exists
        output_dir = f"results/direction_analysis/{arch}_{loss}_{opt}"
        csv_file = f"{output_dir}/direction_summary_{arch}_{loss}_{opt}.csv"
        
        if os.path.exists(csv_file):
            print(f"\n[{i}/{total}] EXISTS: {arch} + {loss} + {opt}")
            print(f"  Output: {output_dir}")
            successful += 1
            continue
        
        print(f"\n[{i}/{total}] RUNNING: {arch} + {loss} + {opt}")
        print(f"  Model: {model_path}")
        
        # Run gradient direction analysis
        try:
            cmd = [
                sys.executable,
                "scripts/analysis/gradient_direction_analysis.py",
                "--architecture", arch,
                "--loss_fn", loss,
                "--optimizer", opt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout
                cwd=os.getcwd()  # Ensure correct working directory
            )
            
            # Check if output was actually created (success indicator)
            if os.path.exists(csv_file):
                print(f"  ✓ SUCCESS (output file created)")
                print(f"  Output: {output_dir}")
                successful += 1
            elif result.returncode == 0:
                print(f"  ✓ SUCCESS")
                print(f"  Output: {output_dir}")
                successful += 1
            else:
                print(f"  ✗ FAILED")
                # Only show stdout as actual error (stderr contains progress bar)
                error_msg = result.stdout if result.stdout else "Unknown error"
                print(f"  Error (first 300 chars): {error_msg[:300]}")
                failed.append((arch, loss, opt, error_msg[:500]))
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (exceeded 20 minutes)")
            failed.append((arch, loss, opt, "Timeout - exceeded 20 minutes"))
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed.append((arch, loss, opt, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models: {total}")
    print(f"Successful: {successful}")
    print(f"Skipped (model not found): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    
    if skipped:
        print("\nSkipped models:")
        for arch, loss, opt in skipped:
            print(f"  - {arch}_{loss}_{opt}")
    
    if failed:
        print("\nFailed models:")
        for arch, loss, opt, error in failed:
            print(f"  - {arch}_{loss}_{opt}")
            print(f"    Error: {error[:100]}")
    
    print("\nAll results saved to: results/direction_analysis/")

if __name__ == "__main__":
    main()
