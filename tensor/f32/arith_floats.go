package tensorf32

import "github.com/chewxy/math32"

// This file deals with floats

func (t *Tensor) HasNaN() bool {
	for _, d := range t.data {
		if math32.IsNaN(d) {
			return true
		}
	}
	return false
}
