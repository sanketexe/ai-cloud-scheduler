"""
Integration test for webhook system
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from core.webhook_test_framework import run_webhook_integration_tests


async def main():
    """Run webhook integration tests"""
    print("Starting webhook integration tests...")
    
    try:
        results = await run_webhook_integration_tests()
        
        print("\n" + "="*60)
        print("WEBHOOK INTEGRATION TEST RESULTS")
        print("="*60)
        
        print(f"Test Suite Started: {results['test_suite_started']}")
        print(f"Test Suite Completed: {results['test_suite_completed']}")
        print(f"Overall Success: {results['overall_success']}")
        
        print("\nIndividual Test Results:")
        print("-" * 40)
        
        for test_name, test_result in results['tests'].items():
            status = "✅ PASS" if test_result.get('success', False) else "❌ FAIL"
            print(f"{test_name}: {status}")
            
            # Print additional details for failed tests
            if not test_result.get('success', False):
                for key, value in test_result.items():
                    if key != 'success':
                        print(f"  {key}: {value}")
        
        if 'error' in results:
            print(f"\nTest Suite Error: {results['error']}")
        
        print("\n" + "="*60)
        
        return results['overall_success']
        
    except Exception as e:
        print(f"Error running webhook integration tests: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)