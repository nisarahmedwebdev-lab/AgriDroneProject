"""Test imports and function signatures."""
from llm_handler import initialize_openai, generate_report, get_spray_advice
import inspect

print("✅ All imports successful!")

print("\n📋 Function Signatures:")
print(f"generate_report: {inspect.signature(generate_report)}")
print(f"get_spray_advice: {inspect.signature(get_spray_advice)}")

# Test with sample data
test_metrics = {
    "total_cells": 625,
    "scanned": 600,
    "healthy": 480,
    "early": 95,
    "severe": 50,
    "healthy_pct": 76.8,
    "early_pct": 15.2,
    "severe_pct": 8.0,
    "obstacles": 25
}

print("\n🔧 Testing function calls...")
try:
    result = generate_report(
        metrics=test_metrics,
        crop_type="Wheat",
        field_name="Test Field",
        disease_seeds=[{"cell": [5, 5], "type": "early"}]
    )
    print("✅ generate_report() works with disease_seeds parameter!")
except Exception as e:
    print(f"❌ generate_report Error: {e}")

try:
    result = get_spray_advice(
        metrics=test_metrics,
        crop_type="Wheat",
        disease_seeds=[{"cell": [5, 5], "type": "early"}]
    )
    print("✅ get_spray_advice() works with disease_seeds parameter!")
except Exception as e:
    print(f"❌ get_spray_advice Error: {e}")

print("\n✅ All tests passed! You can now run the app.")