// +build !fastmath

package gorgonia

// this file holds the non-hacky version of anything that is in the math_fast.go file

import "math"

// SetOptimizationLevel sets the fast math optimization level. By default, fast math is turned off,
// and this function is a no-op.
//
// Use the `fastmath` build tag to use fast math
func SetOptimizationLevel(i int) {}

func _inversef32(x float32) float32 { return float32(1) / x }
func _inversef64(x float64) float64 { return float64(1) / x }

func _tanhf32(x float32) float32 { return float32(math.Tanh(float64(x))) }
func _tanhf64(x float64) float64 { return math.Tanh(x) }
