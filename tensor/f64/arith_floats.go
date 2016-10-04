package tensorf64

import (
	"math"

	"github.com/chewxy/gorgonia/tensor/types"
)

// This file deals with floats

func (t *Tensor) HasNaN() bool {
	if t.IsView() {
		iter := types.NewFlatIterator(t.AP)

		for i, err := iter.Next(); err == nil; i, err = iter.Next() {
			if math.IsNaN(t.data[i]) {
				return true
			}
		}
		return false
	}

	for _, d := range t.data {
		if math.IsNaN(d) {
			return true
		}
	}
	return false
}

func (t *Tensor) HasInf() bool {
	if t.IsView() {
		iter := types.NewFlatIterator(t.AP)

		for i, err := iter.Next(); err == nil; i, err = iter.Next() {
			if math.IsInf(t.data[i], 0) {
				return true
			}
		}
		return false
	}

	for _, d := range t.data {
		if math.IsInf(d, 0) {
			return true
		}
	}
	return false
}
