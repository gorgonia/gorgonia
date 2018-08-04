// +build !fastmath

package gorgonia

// this file holds the non-hacky version of anything that is in the math_fast.go file

import (
	"math"

	"github.com/chewxy/math32"
)

// SetOptimizationLevel sets the fast math optimization level. By default, fast math is turned off,
// and this function is a no-op.
//
// Use the `fastmath` build tag to use fast math
func SetOptimizationLevel(i int) {}

func _inversef32(x float32) float32 { return float32(1) / x }
func _inversef64(x float64) float64 { return float64(1) / x }

func _tanhf32(x float32) float32 { return float32(math.Tanh(float64(x))) }
func _tanhf64(x float64) float64 { return math.Tanh(x) }

func _sigmoidf64(x float64) float64 {
	if x < -709 {
		return 0
	}
	if x > 19 {
		return 1
	}

	return 1.0 / (1.0 + math.Exp(-x))
}

func _sigmoidf32(x float32) float32 {
	if x < -88 {
		return 0
	}
	if x > 15 {
		return 1
	}
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func _inverseSqrtf64(x float64) float64 {
	return 1 / math.Sqrt(x)
}

func _inverseSqrtf32(x float32) float32 {
	return 1 / math32.Sqrt(x)
}
