#!/usr/bin/env python3
"""Create data list files for training with individual files"""

import argparse
import os
import json

def create_data_lists(src_dir, output_dir):
    """Create data list files pointing to directories or index files
    
    Args:
        src_dir: Directory containing processed audio files
        output_dir: Directory to save list files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Create a list pointing to the source directory
    with open(os.path.join(output_dir, 'data.list'), 'w') as f:
        f.write(src_dir + '\n')
    
    # Option 2: If you have an index file, point to it
    index_file = os.path.join(src_dir, 'data_index.json')
    if os.path.exists(index_file):
        with open(os.path.join(output_dir, 'data_index.list'), 'w') as f:
            f.write(index_file + '\n')
    
    print(f"Created data lists in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True,
                       help='Source directory with processed files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for list files')
    args = parser.parse_args()
    
    create_data_lists(args.src_dir, args.output_dir)