package gorgonia

import (
	"math"

	"github.com/chewxy/math32"
)

// functions in this file are functions that do not have an optimized/hacked up version
// typically I'd have done some benchmark vs some hacked up version and determined that the default implementation is indeed superior

/* UNARY OPS */
func _signf32(x float32) float32 {
	if math32.Signbit(x) {
		return float32(-1.0)
	}
	return 1
}

func _signf64(x float64) float64 {
	if math.Signbit(x) {
		return -1.0
	}
	return 1
}

func _squaref64(x float64) float64 { return x * x }
func _squaref32(x float32) float32 { return x * x }

func _cubef64(x float64) float64 { return x * x * x }
func _cubef32(x float32) float32 { return x * x * x }

func _negf32(x float32) float32 { return -x }
func _negf64(x float64) float64 { return -x }

/* TODO: write optimized versions of these */

// bounds acquired with this:
/*
func main() {
	var buf bytes.Buffer
	for i := -1000; i < 1000; i++ {
		res := math.Log1p(math.Exp(float64(i)))
		fmt.Fprintf(&buf, "%d\t%v\n", i, res)
	}
	fmt.Println(buf.String())
}
*/
// I chose 16 because from 17 onwards to 709, its pretty much  returns x (with some stupid small decimals)
func _softplusf64(x float64) float64 {
	if x < -708 {
		return 0
	}
	if x > 16 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

func _softplusf32(x float32) float32 {
	if x < -103 {
		return 0
	}
	if x > 14 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}
