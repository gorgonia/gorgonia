package tensorf32

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/math32"
)

// This file deals with floats

func (t *Tensor) HasNaN() bool {
	if t.IsView() {
		iter := types.NewFlatIterator(t.AP)

		for i, err := iter.Next(); err == nil; i, err = iter.Next() {
			if math32.IsNaN(t.data[i]) {
				return true
			}
		}
		return false
	}

	for _, d := range t.data {
		if math32.IsNaN(d) {
			return true
		}
	}
	return false
}

func (t *Tensor) HasInf() bool {
	if t.IsView() {
		iter := types.NewFlatIterator(t.AP)

		for i, err := iter.Next(); err == nil; i, err = iter.Next() {
			if math32.IsInf(t.data[i], 0) {
				return true
			}
		}
		return false
	}

	for _, d := range t.data {
		if math32.IsInf(d, 0) {
			return true
		}
	}
	return false
}
