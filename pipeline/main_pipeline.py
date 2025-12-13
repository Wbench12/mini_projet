"""Main Pipeline - Orchestrates all dataset pipelines"""

import sys
import os

# Add project root to path so we can import pipeline modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.drug_discovery_pipeline import run_pipeline as run_drug_discovery
from pipeline.ct_tox_pipeline import run_pipeline as run_ct_tox
from pipeline.tox21_pipeline import run_pipeline as run_tox21


def main():
    """Run all pipelines or specific ones based on arguments."""
    print("="*60)
    print("🔬 QSAR SEMI-SUPERVISED PIPELINE")
    print("   Orchestrating All Dataset Pipelines")
    print("="*60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        
        if dataset == 'drug_discovery':
            print("\n▶️ Running Drug Discovery Pipeline only...")
            run_drug_discovery()
        elif dataset == 'ct_tox':
            print("\n▶️ Running CT_TOX Pipeline only...")
            run_ct_tox()
        elif dataset == 'tox21':
            print("\n▶️ Running TOX21 Pipeline only...")
            run_tox21()
        else:
            print(f"❌ Unknown dataset: {dataset}")
            print("Usage: python pipeline/main_pipeline.py [drug_discovery|ct_tox|tox21]")
            print("   Or run without arguments to process all datasets")
            return
    else:
        # Run all pipelines
        print("\n▶️ Running All Pipelines...")
        
        print("\n" + "="*60)
        run_drug_discovery()
        
        print("\n" + "="*60)
        run_ct_tox()
        
        print("\n" + "="*60)
        run_tox21()
    
    print("\n" + "="*60)
    print("✅ ALL PIPELINES COMPLETE")
    print("   ✓ Enhanced data processed")
    print("   ✓ Processed data saved to data/processed/")
    print("   ✓ Models trained and saved to app/models/")
    print("   ✓ Scalers saved with models for inference")
    print("   Restart Uvicorn to apply changes.")
    print("="*60)


if __name__ == "__main__":
    main()
