import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Test Molecules
# 1. Aspirin (Known Safe & Approved)
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
# 2. Thalidomide (Known Toxic/Withdrawn)
THALIDOMIDE = "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"
# 3. Ibuprofen (Safe)
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def test_models_info():
    """Test 1: Check if models are loaded"""
    print_section("TEST 1: Get Models Info")
    url = f"{API_URL}/qsar/models-info"
    
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            
            # Simple assertion
            if data['fda_model']['status'] == 'Loaded' and data['toxicity_model']['status'] == 'Loaded':
                print("\n✅ SUCCESS: Models are loaded and ready.")
            else:
                print("\n⚠️ WARNING: Models report as 'Not Loaded'. Check logs.")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Is uvicorn running?")
        sys.exit(1)

def test_single_prediction():
    """Test 2: Predict for a single molecule"""
    print_section("TEST 2: Single Molecule Prediction")
    url = f"{API_URL}/qsar/predict"
    
    payload = {"smiles": ASPIRIN}
    print(f"Sending SMILES: {ASPIRIN} (Aspirin)")
    
    response = requests.post(url, json=payload, headers=HEADERS)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response ---")
        print(f"Molecule: {data['smiles']}")
        print(f"FDA Prediction: {data['fda_approval']['prediction_label']} ({data['fda_approval']['confidence']:.2%})")
        print(f"Toxicity Risk:  {data['toxicity']['prediction_label']} ({data['toxicity']['confidence']:.2%})")
        print(f"Recommendation: {data['recommendation']}")
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_batch_prediction():
    """Test 3: Predict for multiple molecules"""
    print_section("TEST 3: Batch Prediction")
    url = f"{API_URL}/qsar/predict-batch"
    
    batch_payload = {
        "molecules": [
            {"smiles": ASPIRIN},
            {"smiles": THALIDOMIDE},
            {"smiles": "INVALID_SMILES_STRING_TEST"} # Intentional error to test handling
        ]
    }
    
    print(f"Sending batch of {len(batch_payload['molecules'])} molecules...")
    
    response = requests.post(url, json=batch_payload, headers=HEADERS)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal Processed: {data['total']}")
        print(f"Successful:      {data['successful']}")
        print(f"Failed:          {data['failed']}")
        
        print("\n--- Results ---")
        for res in data['predictions']:
            print(f"[{res['molecular_properties']['molecular_weight']:.1f} g/mol] {res['fda_approval']['prediction_label']} | {res['toxicity']['prediction_label']}")
            
        if data['errors']:
            print("\n--- Errors (Expected) ---")
            for err in data['errors']:
                print(f"Item {err['index']}: {err['error']}")
                
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_get_prediction():
    """Test 4: GET convenience endpoint"""
    print_section("TEST 4: GET Request with Query Parameter")
    url = f"{API_URL}/qsar/predict-smiles"
    params = {"smiles": IBUPROFEN}
    
    print(f"Requesting: {url}?smiles={IBUPROFEN}")
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    print(f"🧪 DRUG DISCOVERY API TEST SUITE")
    print(f"Target: {API_URL}")
    
    test_models_info()
    test_single_prediction()
    test_batch_prediction()
    test_get_prediction()
    
    print_section("✅ ALL TESTS COMPLETED")
