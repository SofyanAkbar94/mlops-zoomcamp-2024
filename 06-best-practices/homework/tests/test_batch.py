import pytest
from batch_refactor import main  # Import the main function from batch_refactor.py

def test_main():
    year = 2023
    month = 3
    main(year, month)
    # You can add assertions here to check the output