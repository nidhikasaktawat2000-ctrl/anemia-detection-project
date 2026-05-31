"""Validate dataset labels against WHO hemoglobin standards.

WHO STANDARDS:
- Non-anemic female: Hb >= 12.0 g/dL
- Non-anemic male:   Hb >= 13.0 g/dL
- Anemic female:     Hb < 12.0 g/dL
- Anemic male:       Hb < 13.0 g/dL
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

def validate_dataset(data_dir: str = "data", metadata_path: str = None):
    """Verify labels and check for corrupt images."""
    
    report = {
        "summary": {
            "total_images": 0,
            "corrupt_images": 0,
            "label_mismatches": 0,
        },
        "anemia_folder": {
            "count": 0,
            "corrupt": 0,
            "sizes": []
        },
        "non_anemia_folder": {
            "count": 0,
            "corrupt": 0,
            "sizes": []
        },
        "potential_label_errors": []
    }
    
    # Load metadata if available
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print("\n" + "="*70)
    print("DATASET LABEL VALIDATION (WHO HEMOGLOBIN STANDARDS)")
    print("="*70)
    
    for label_name in ["anemia", "non_anemia"]:
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"\n⚠️  Folder not found: {label_dir}")
            continue
        
        print(f"\n[{label_name.upper()}]")
        
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            if not os.path.isfile(fpath):
                continue
            
            # Check if image is readable
            try:
                with Image.open(fpath) as img:
                    img.load()
                    w, h = img.size
                    report[f"{label_name}_folder"]["count"] += 1
                    report[f"{label_name}_folder"]["sizes"].append((w, h))
                    report["summary"]["total_images"] += 1
            except Exception as e:
                report[f"{label_name}_folder"]["corrupt"] += 1
                report["summary"]["corrupt_images"] += 1
                print(f"  ❌ CORRUPT: {fname} - {e}")
        
        # Validate against metadata (Hb values)
        if metadata:
            for fname, record in metadata.items():
                if fname not in [f for f in os.listdir(label_dir)]:
                    continue
                
                hb = record.get("hemoglobin")
                gender = record.get("gender", "Female").lower()
                
                if hb is None:
                    continue
                
                # Check WHO standards
                hb_threshold = 12.0 if gender == "female" else 13.0
                is_anemic_hb = hb < hb_threshold
                is_anemic_label = label_name == "anemia"
                
                if is_anemic_hb != is_anemic_label:
                    report["potential_label_errors"].append({
                        "file": fname,
                        "labeled_as": label_name,
                        "hb_value": hb,
                        "gender": gender,
                        "who_indicates": "ANEMIC" if is_anemic_hb else "NORMAL",
                        "severity": "CRITICAL" if abs(hb - hb_threshold) > 2.0 else "MINOR"
                    })
                    report["summary"]["label_mismatches"] += 1
    
    # Print summary
    print(f"\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"\nTotal images scanned: {report['summary']['total_images']}")
    print(f"Corrupt/unreadable:  {report['summary']['corrupt_images']}")
    print(f"\nAnemia folder:")
    print(f"  ✓ Valid images: {report['anemia_folder']['count']}")
    print(f"  ❌ Corrupt:     {report['anemia_folder']['corrupt']}")
    print(f"\nNon-anemia folder:")
    print(f"  ✓ Valid images: {report['non_anemia_folder']['count']}")
    print(f"  ❌ Corrupt:     {report['non_anemia_folder']['corrupt']}")
    
    if report['summary']['label_mismatches'] > 0:
        print(f"\n⚠️  POTENTIAL LABEL ERRORS: {report['summary']['label_mismatches']}")
        for error in report["potential_label_errors"]:
            print(f"\n  File: {error['file']}")
            print(f"    Labeled as:    {error['labeled_as']}")
            print(f"    Hb value:      {error['hb_value']} g/dL")
            print(f"    Gender:        {error['gender']}")
            print(f"    WHO indicates: {error['who_indicates']} (Severity: {error['severity']})")
    else:
        print(f"\n✅ All labels match WHO hemoglobin standards")
    
    # Save report
    report_path = "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    validate_dataset("data")
