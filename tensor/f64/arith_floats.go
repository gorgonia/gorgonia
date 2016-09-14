package tensorf64

import "math"

// This file deals with floats

func (t *Tensor) HasNaN() bool {
	for _, d := range t.data {
		if math.IsNaN(d) {
			return true
		}
	}
	return false
}
