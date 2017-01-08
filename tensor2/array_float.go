package tensor

import (
	"math"

	"github.com/chewxy/math32"
)

func (a f64s) HasNaN() bool {
	for _, v := range a {
		if math.IsNaN(v) {
			return true
		}
	}
	return false
}

func (a f32s) HasNaN() bool {
	for _, v := range a {
		if math32.IsNaN(v) {
			return true
		}
	}
	return false
}

func (a f64s) HasInf() bool {
	for _, v := range a {
		if math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

func (a f32s) HasInf() bool {
	for _, v := range a {
		if math32.IsInf(v, 0) {
			return true
		}
	}
	return false
}
