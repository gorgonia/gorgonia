package tensori

import "github.com/chewxy/gorgonia/tensor/types"

/* This file deals with the arg methods */

func (t *Tensor) Argmax(axis int) (retVal *Tensor, err error) {
	if axis == types.AllAxes {
		// retVal = argmax(t.data)
		// retVal = []int{argmax(t.data)}
		retVal = NewTensor(AsScalar(argmax(t.data)))
		return
	}
	if axis >= len(t.Shape()) {
		err = types.DimMismatchErr(len(t.Shape()), axis)
		return
	}

	var indices []int
	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}

	// be a good citizen - borrow and return, since we're only using this AP to figure out the moves
	newAP, _, err := t.AP.T(axes...)
	if _, ok := err.(NoOpError); !ok && err != nil {
		return
	} else if ok {
		err = nil // reset errs
		newAP = t.AP.Clone()
	}
	defer types.ReturnAP(newAP)

	lastSize := newAP.Shape()[len(newAP.Shape())-1]
	split := len(t.data) / lastSize
	start := 0
	for i := 0; i < split; i++ {
		max := argmax(t.data[start : start+lastSize])

		start += lastSize
		indices = append(indices, max)
	}

	newShape := newAP.Shape().Clone()
	newShape = newShape[:len(newShape)-1]
	defer types.ReturnInts(newShape)

	retT := NewTensor(WithShape(newShape...), WithBacking(indices))
	retVal = retT
	return
}
