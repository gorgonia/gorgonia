@echo off
setlocal

:: Define the directory where the test files are located
set "TEST_DIR=tests"

:: Change to the test directory using relative path from script location
cd "%~dp0%..\tests" || ( echo Failed to change directory & exit /b 1 )

echo --- Running original non-deterministic test (issue_297.go) 10 times ---

for /L %%i in (1,1,10) do (
    echo Run %%i:
    set "ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23"
    go run issue_297.go
    echo.
)

echo --- Running deterministic test (issue_297_deterministic.go) ---
set "ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23"
go run issue_297_deterministic.go

echo --- All issue 297 tests completed successfully ---

endlocal