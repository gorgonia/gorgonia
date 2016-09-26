package tensorf64

import (
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
)

/* This file deals with the arg methods */

func (t *Tensor) Argmax(axis int) (retVal *ti.Tensor, err error) {
	if axis == types.AllAxes {
		retVal = ti.NewTensor(ti.AsScalar(argmax(t.data)))
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

	tmp := make([]float64, 0, lastSize)
	it := types.NewFlatIterator(newAP)
	for next, err := it.Next(); err == nil; next, err = it.Next() {
		tmp = append(tmp, t.data[next])

		if len(tmp) == lastSize {
			am := argmax(tmp)
			indices = append(indices, am)

			// reset
			tmp = tmp[:0]
		}
	}

	newShape := newAP.Shape().Clone()
	newShape = newShape[:len(newShape)-1]
	defer types.ReturnInts(newShape)

	retT := ti.NewTensor(ti.WithShape(newShape...), ti.WithBacking(indices))
	retVal = retT
	return

	return
}

func (t *Tensor) Argmin(axis int) (retVal *ti.Tensor, err error) {
	if axis == types.AllAxes {
		retVal = ti.NewTensor(ti.AsScalar(argmin(t.data)))
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

	tmp := make([]float64, 0, lastSize)
	it := types.NewFlatIterator(newAP)
	for next, err := it.Next(); err == nil; next, err = it.Next() {
		tmp = append(tmp, t.data[next])

		if len(tmp) == lastSize {
			am := argmin(tmp)
			indices = append(indices, am)

			// reset
			tmp = tmp[:0]
		}
	}

	newShape := newAP.Shape().Clone()
	newShape = newShape[:len(newShape)-1]
	defer types.ReturnInts(newShape)

	retT := ti.NewTensor(ti.WithShape(newShape...), ti.WithBacking(indices))
	retVal = retT
	return

	return
}
