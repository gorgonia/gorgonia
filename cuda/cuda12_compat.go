//go:build cuda && !darwin && !arm64

package cuda

// This file ensures that the CUDA 12.6 compatibility wrapper is compiled and
// linked with the Go code. The wrapper provides missing _v2 function symbols
// that are referenced by gorgonia.org/cu but not exported by the CUDA library.

// The cuda12_compat.c file will be automatically compiled by CGO when this
// package is built. It provides the missing _v2 symbols.
