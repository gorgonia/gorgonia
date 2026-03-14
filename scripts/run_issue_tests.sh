#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the directory where the test files are located
TEST_DIR="tests"

echo "--- Running original non-deterministic test (issue_297.go) 10 times ---"
cd "$TEST_DIR"

for i in {1..10}; do
    echo "Run $i:"
    # Execute the non-deterministic test
    ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23 go run issue_297.go
    echo "" # Add a newline for better readability between runs
done

echo "--- Running deterministic test (issue_297_deterministic.go) ---"
# Execute the deterministic test once
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23 go run issue_297_deterministic.go

echo "--- All issue 297 tests completed successfully ---"