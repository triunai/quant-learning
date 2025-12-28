"""Test runner script for quant-learning project."""
import sys
import pytest

def main():
    """Run pytest with appropriate arguments."""
    args = [
        'tests/',
        '-v',
        '--tb=short',
        '--strict-markers',
        '-m', 'not slow',  # Skip slow tests by default
    ]
    
    # Add any command line args
    args.extend(sys.argv[1:])
    
    return pytest.main(args)

if __name__ == '__main__':
    sys.exit(main())