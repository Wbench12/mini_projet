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
# 4. Metformin (Approved, diabetes drug)
METFORMIN = "CN(C)C(=N)N=C(N)N"

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
            
            # Check new structure
            if data.get('drug_model', {}).get('status') == 'Loaded' and \
               data.get('toxicity_model', {}).get('status') == 'Loaded':
                print("\n✅ SUCCESS: Models are loaded and ready.")
            else:
                print("\n⚠️ WARNING: Models report as 'Not Loaded'. Check logs.")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Is uvicorn running?")
        sys.exit(1)

def test_drug_prediction():
    """Test 2: Predict drug approval only"""
    print_section("TEST 2: Drug Approval Prediction (Single)")
    url = f"{API_URL}/qsar/drug-predict"
    
    payload = {"smiles": ASPIRIN}
    print(f"Sending SMILES: {ASPIRIN} (Aspirin)")
    
    response = requests.post(url, json=payload, headers=HEADERS)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response ---")
        print(f"Molecule: {data['smiles']}")
        print(f"Canonical SMILES: {data['canonical_smiles']}")
        print(f"FDA Prediction: {data['prediction']['prediction_label']}")
        print(f"Confidence: {data['prediction']['confidence']:.2%}")
        print(f"Probabilities: {data['prediction']['probabilities']}")
        if data.get('molecular_properties'):
            print(f"Molecular Weight: {data['molecular_properties']['molecular_weight']:.2f} g/mol")
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_toxicity_prediction():
    """Test 3: Predict toxicity only"""
    print_section("TEST 3: Toxicity Prediction (Single)")
    url = f"{API_URL}/qsar/toxicity-predict"
    
    payload = {"smiles": THALIDOMIDE}
    print(f"Sending SMILES: {THALIDOMIDE} (Thalidomide)")
    
    response = requests.post(url, json=payload, headers=HEADERS)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response ---")
        print(f"Molecule: {data['smiles']}")
        print(f"Canonical SMILES: {data['canonical_smiles']}")
        print(f"Toxicity Prediction: {data['prediction']['prediction_label']}")
        print(f"Confidence: {data['prediction']['confidence']:.2%}")
        print(f"Probabilities: {data['prediction']['probabilities']}")
        if data.get('molecular_properties'):
            print(f"Molecular Weight: {data['molecular_properties']['molecular_weight']:.2f} g/mol")
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_complete_prediction():
    """Test 4: Predict both drug approval and toxicity"""
    print_section("TEST 4: Complete Prediction (Both Drug & Toxicity)")
    url = f"{API_URL}/qsar/predict"
    
    payload = {"smiles": IBUPROFEN}
    print(f"Sending SMILES: {IBUPROFEN} (Ibuprofen)")
    
    response = requests.post(url, json=payload, headers=HEADERS)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response ---")
        print(f"Molecule: {data['smiles']}")
        print(f"FDA Prediction: {data['drug_approval']['prediction_label']} ({data['drug_approval']['confidence']:.2%})")
        print(f"Toxicity Risk:  {data['toxicity']['prediction_label']} ({data['toxicity']['confidence']:.2%})")
        print(f"Recommendation: {data['recommendation']}")
        print(f"Risk Level: {data['risk_level']}")
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_drug_batch_prediction():
    """Test 5: Batch drug approval predictions"""
    print_section("TEST 5: Batch Drug Approval Prediction")
    url = f"{API_URL}/qsar/drug-predict-batch"
    
    batch_payload = {
        "molecules": [
            {"smiles": ASPIRIN},
            {"smiles": IBUPROFEN},
            {"smiles": METFORMIN},
            {"smiles": "INVALID_SMILES_STRING_TEST"}  # Intentional error
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
            pred = res['prediction']
            print(f"[{res['smiles'][:30]:30s}] {pred['prediction_label']:20s} (Conf: {pred['confidence']:.2%})")
            
        if data['errors']:
            print("\n--- Errors (Expected) ---")
            for err in data['errors']:
                print(f"Item {err['index']}: {err['error']}")
                
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_toxicity_batch_prediction():
    """Test 6: Batch toxicity predictions"""
    print_section("TEST 6: Batch Toxicity Prediction")
    url = f"{API_URL}/qsar/toxicity-predict-batch"
    
    batch_payload = {
        "molecules": [
            {"smiles": ASPIRIN},
            {"smiles": THALIDOMIDE},
            {"smiles": IBUPROFEN},
            {"smiles": "INVALID_SMILES_STRING_TEST"}  # Intentional error
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
            pred = res['prediction']
            print(f"[{res['smiles'][:30]:30s}] {pred['prediction_label']:20s} (Conf: {pred['confidence']:.2%})")
            
        if data['errors']:
            print("\n--- Errors (Expected) ---")
            for err in data['errors']:
                print(f"Item {err['index']}: {err['error']}")
                
        print("\n✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_drug_get_prediction():
    """Test 7: GET convenience endpoint for drug prediction"""
    print_section("TEST 7: GET Request - Drug Prediction")
    url = f"{API_URL}/qsar/drug-predict-smiles"
    params = {"smiles": METFORMIN}
    
    print(f"Requesting: {url}?smiles={METFORMIN}")
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"FDA Prediction: {data['prediction']['prediction_label']}")
        print(f"Confidence: {data['prediction']['confidence']:.2%}")
        print("✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_toxicity_get_prediction():
    """Test 8: GET convenience endpoint for toxicity prediction"""
    print_section("TEST 8: GET Request - Toxicity Prediction")
    url = f"{API_URL}/qsar/toxicity-predict-smiles"
    params = {"smiles": ASPIRIN}
    
    print(f"Requesting: {url}?smiles={ASPIRIN}")
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Toxicity Prediction: {data['prediction']['prediction_label']}")
        print(f"Confidence: {data['prediction']['confidence']:.2%}")
        print("✅ SUCCESS")
    else:
        print(f"❌ Error: {response.text}")

def test_multiple_molecules():
    """Test 9: Test multiple molecules with both endpoints"""
    print_section("TEST 9: Multiple Molecules Comparison")
    
    molecules = [
        ("Aspirin", ASPIRIN),
        ("Ibuprofen", IBUPROFEN),
        ("Thalidomide", THALIDOMIDE),
        ("Metformin", METFORMIN)
    ]
    
    print(f"\n{'Molecule':<15} {'Drug Approval':<20} {'Toxicity':<20}")
    print("-" * 60)
    
    for name, smiles in molecules:
        # Drug prediction
        drug_url = f"{API_URL}/qsar/drug-predict"
        drug_resp = requests.post(drug_url, json={"smiles": smiles}, headers=HEADERS)
        
        # Toxicity prediction
        tox_url = f"{API_URL}/qsar/toxicity-predict"
        tox_resp = requests.post(tox_url, json={"smiles": smiles}, headers=HEADERS)
        
        if drug_resp.status_code == 200 and tox_resp.status_code == 200:
            drug_data = drug_resp.json()
            tox_data = tox_resp.json()
            
            drug_label = drug_data['prediction']['prediction_label']
            tox_label = tox_data['prediction']['prediction_label']
            
            print(f"{name:<15} {drug_label:<20} {tox_label:<20}")
        else:
            print(f"{name:<15} ERROR                ERROR")
    
    print("\n✅ SUCCESS")

if __name__ == "__main__":
    print(f"🧪 DRUG DISCOVERY API TEST SUITE")
    print(f"Target: {API_URL}")
    
    # Run all tests
    test_models_info()
    test_drug_prediction()
    test_toxicity_prediction()
    test_complete_prediction()
    test_drug_batch_prediction()
    test_toxicity_batch_prediction()
    test_drug_get_prediction()
    test_toxicity_get_prediction()
    test_multiple_molecules()
    
    print_section("✅ ALL TESTS COMPLETED")
    print("\nSummary:")
    print("  ✓ Model info endpoint")
    print("  ✓ Drug approval prediction (POST)")
    print("  ✓ Toxicity prediction (POST)")
    print("  ✓ Complete prediction (both)")
    print("  ✓ Batch drug predictions")
    print("  ✓ Batch toxicity predictions")
    print("  ✓ GET endpoints (drug & toxicity)")
    print("  ✓ Multiple molecules comparison")
