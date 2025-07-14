#!/usr/bin/env python3
"""
Test script for the new preprocessing functionality
"""

import os
import tempfile
import jsonlines
from src.DataProcess import DataProcessor

def test_data_processor():
    """Test the DataProcessor with different input formats"""
    
    # Create test data
    test_data = [
        # Format 1: Single column JSON
        '{"data_id": "test_001", "text": "This is test text 1"}',
        '{"data_id": "test_002", "text": "This is test text 2"}',
        
        # Format 2: Two column TSV
        'doc_001\t{"text": "This is test text 3"}',
        'doc_002\t{"text": "This is test text 4"}',
        
        # Format 3: Two column TSV with additional fields
        'article_001\t{"text": "This is test text 5", "source": "web"}',
        'article_002\t{"text": "This is test text 6", "metadata": {"category": "test"}}',
        
        # Edge cases
        '{"data_id": "json_priority", "text": "JSON data_id should take priority"}',
        'tsv_id\t{"data_id": "json_priority", "text": "JSON data_id should take priority"}',
        
        # Invalid lines (should be skipped)
        'invalid line without proper format',
        '{"invalid": "missing text field"}',
        'id\t{"invalid": "missing text field"}',
    ]
    
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for line in test_data:
            f.write(line + '\n')
        input_file = f.name
    
    # Create temporary output directory
    output_dir = tempfile.mkdtemp()
    
    try:
        # Initialize DataProcessor
        processor = DataProcessor(max_workers=2, batch_size=3)
        
        # Process the file
        output_file = os.path.join(output_dir, 'test_output.jsonl')
        processor.process_file(input_file, output_file, 0)
        
        # Read and verify results
        results = []
        with jsonlines.open(output_file, 'r') as reader:
            for obj in reader:
                results.append(obj)
        
        print(f"Processed {len(results)} valid records")
        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"{i}. data_id: {result['data_id']}, content: {result['content'][:50]}...")
        
        # Verify expected results
        expected_ids = [
            "test_001", "test_002", 
            "doc_001", "doc_002",
            "article_001", "article_002",
            "json_priority", "json_priority"  # This should appear twice due to edge case
        ]
        
        actual_ids = [r['data_id'] for r in results]
        print(f"\nExpected IDs: {expected_ids}")
        print(f"Actual IDs: {actual_ids}")
        
        # Check if all expected IDs are present
        missing_ids = set(expected_ids) - set(actual_ids)
        extra_ids = set(actual_ids) - set(expected_ids)
        
        if missing_ids:
            print(f"Missing IDs: {missing_ids}")
        if extra_ids:
            print(f"Extra IDs: {extra_ids}")
        
        if not missing_ids and not extra_ids:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
    finally:
        # Clean up
        os.unlink(input_file)
        for file in os.listdir(output_dir):
            os.unlink(os.path.join(output_dir, file))
        os.rmdir(output_dir)

if __name__ == "__main__":
    test_data_processor() 