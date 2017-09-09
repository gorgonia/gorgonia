package tensor

import "github.com/pkg/errors"

func (e StdEng) Repeat(t Tensor, axis int, repeats ...int) (Tensor, error) {
	switch tt := t.(type) {
	case DenseTensor:
		return e.denseRepeat(tt, axis, repeats)
	default:
		return nil, errors.Errorf("NYI")
	}
}

func (StdEng) denseRepeat(t DenseTensor, axis int, repeats []int) (retVal DenseTensor, err error) {
	var newShape Shape
	var size int
	if newShape, repeats, size, err = t.Shape().Repeat(axis, repeats...); err != nil {
		return nil, errors.Wrap(err, "Unable to get repeated shape")
	}

	if axis == AllAxes {
		axis = 0
	}

	d := recycledDense(t.Dtype(), newShape)

	var outers int
	if t.IsScalar() {
		outers = 1
	} else {
		outers = ProdInts(t.Shape()[0:axis])
		if outers == 0 {
			outers = 1
		}
	}

	var stride, newStride int
	if newShape.IsVector() || t.IsVector() {
		stride = 1 // special case because CalcStrides() will return []int{1} as the strides for a vector
	} else {
		stride = t.ostrides()[axis]
	}

	if newShape.IsVector() {
		newStride = 1
	} else {
		newStride = d.ostrides()[axis]
	}

	var destStart, srcStart int
	for i := 0; i < outers; i++ {
		for j := 0; j < size; j++ {
			var tmp int
			tmp = repeats[j]

			for k := 0; k < tmp; k++ {
				if srcStart >= t.len() || destStart+stride > d.len() {
					break
				}
				copyDenseSliced(d, destStart, d.len(), t, srcStart, t.len())
				destStart += newStride
			}
			srcStart += stride
		}
	}
	return d, nil
}

func (e StdEng) Concat(t Tensor, axis int, others ...Tensor) (retVal Tensor, err error) {
	switch tt := t.(type) {
	case DenseTensor:
		var denses []DenseTensor
		if denses, err = tensorsToDenseTensors(others); err != nil {
			return nil, errors.Wrap(err, "Concat failed")
		}
		return e.denseConcat(tt, axis, denses)
	default:
		return nil, errors.Errorf("NYI")
	}
}

func (e StdEng) denseConcat(a DenseTensor, axis int, Ts []DenseTensor) (DenseTensor, error) {
	ss := make([]Shape, len(Ts))
	var err error
	var isMasked bool
	for i, T := range Ts {
		ss[i] = T.Shape()
		if mt, ok := T.(MaskedTensor); ok {
			isMasked = isMasked || mt.IsMasked()
		}
	}

	var newShape Shape
	if newShape, err = a.Shape().Concat(axis, ss...); err != nil {
		return nil, errors.Wrap(err, "Unable to find new shape that results from concatenation")
	}

	retVal := recycledDense(a.Dtype(), newShape)
	if isMasked {
		retVal.makeMask()
	}

	all := make([]DenseTensor, len(Ts)+1)
	all[0] = a
	copy(all[1:], Ts)

	// special case
	var start, end int

	for _, T := range all {
		end += T.Shape()[axis]
		slices := make([]Slice, axis+1)
		slices[axis] = makeRS(start, end)

		var v *Dense
		if v, err = sliceDense(retVal, slices...); err != nil {
			return nil, errors.Wrap(err, "Unable to slice DenseTensor while performing denseConcat")
		}

		if v.IsVector() && T.IsMatrix() && axis == 0 {
			v.reshape(v.shape[0], 1)
		}

		if err = assignArray(v, T); err != nil {
			return nil, errors.Wrap(err, "Unable to assignArray in denseConcat")
		}
		start = end
	}

	return retVal, nil
}
