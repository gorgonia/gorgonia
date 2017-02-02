package tensorf64

import (
	"math"

	"github.com/chewxy/gorgonia/tensor/types"
)

func (t *Tensor) multiSVDNorm(rowAxis, colAxis int) (retVal *Tensor, err error) {
	if rowAxis > colAxis {
		rowAxis--
	}
	dims := t.Dims()

	if retVal, err = t.RollAxis(colAxis, dims, true); err != nil {
		return
	}

	if retVal, err = retVal.RollAxis(rowAxis, dims, true); err != nil {
		return
	}

	// manual, since SVD only works on matrices. In the future, this needs to be fixed when gonum's lapack works for float32
	// TODO: SVDFuture
	switch dims {
	case 2:
		retVal, _, _, err = retVal.SVD(false, false)
	case 3:
		toStack := make([]*Tensor, retVal.Shape()[0])
		for i := 0; i < retVal.Shape()[0]; i++ {
			var sliced, ithS *Tensor
			if sliced, err = retVal.Slice(ss(i)); err != nil {
				return
			}

			if ithS, _, _, err = sliced.SVD(false, false); err != nil {
				return
			}

			toStack[i] = ithS
		}

		retVal, err = toStack[0].Stack(0, toStack[1:]...)
		return
	default:
		err = notyetimplemented("multiSVDNorm for dimensions greater than 3")
	}

	return
}

// Norm returns the p-ordered norm of the *Tensor, given the axes.
//
// This implementation is directly adapted from Numpy, which is licenced under a BSD-like licence, and can be found here: https://docs.scipy.org/doc/numpy-1.9.1/license.html
func (t *Tensor) Norm(ord types.NormOrder, axes ...int) (retVal *Tensor, err error) {
	dims := t.Dims()

	// simple case
	if len(axes) == 0 {
		if ord.IsUnordered() || (ord.IsFrobenius() && dims == 2) || (ord == types.Norm(2) && dims == 1) {
			backup := t.AP
			ap := types.BorrowAP(1)
			defer types.ReturnAP(ap)

			ap.Unlock()
			ap.SetShape(t.Size())
			ap.Lock()

			t.AP = ap
			retVal, err = Dot(t, t) // returns a scalar
			retVal.data[0] = math.Sqrt(retVal.data[0])
			t.AP = backup
			return
		}

		axes = make([]int, dims)
		for i := range axes {
			axes[i] = i
		}
	}

	switch len(axes) {
	case 1:
		cloned := t.Clone()
		switch {
		case ord.IsUnordered() || ord == types.Norm(2):
			if retVal, err = PointwiseSquare(cloned); err != nil {
				return
			}

			if retVal, err = retVal.Sum(axes...); err != nil {
				return
			}

			return Sqrt(retVal)
		case ord.IsInf(1):
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			return retVal.Max(axes...)
		case ord.IsInf(-1):
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			return retVal.Min(axes...)
		case ord == types.Norm(0):
			fn := func(x float64) float64 {
				if x != float64(0) {
					return 1
				}
				return 0
			}
			if retVal, err = cloned.Apply(fn); err != nil {
				return
			}
			return retVal.Sum(axes...)
		case ord == types.Norm(1):
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			return retVal.Sum(axes...)
		default:
			fn := func(x float64) float64 {
				return math.Pow(math.Abs(x), float64(ord))
			}
			if retVal, err = cloned.Apply(fn); err != nil {
				return
			}

			if retVal, err = retVal.Sum(axes...); err != nil {
				return
			}

			retVal, err = PointwisePow(retVal, float64(1)/float64(ord))
			return
		}
	case 2:
		rowAxis := axes[0]
		colAxis := axes[1]

		// checks
		if rowAxis < 0 {
			err = types.NewError(types.IndexError, "Row Axis %d is < 0", rowAxis)
			return
		}
		if colAxis < 0 {
			err = types.NewError(types.IndexError, "Col Axis %d is < 0", colAxis)
			return
		}

		if rowAxis == colAxis {
			err = types.NewError(types.AxisError, "Duplicate axes found. Row Axis: %d, Col Axis %d", rowAxis, colAxis)
			return
		}

		cloned := t.Clone()
		switch {
		case ord == types.Norm(2):
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAxis, colAxis); err != nil {
				return
			}
			dims := retVal.Dims()
			return retVal.Max(dims - 1)
		case ord == types.Norm(-2):
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAxis, colAxis); err != nil {
				return
			}
			dims := retVal.Dims()
			return retVal.Min(dims - 1)
		case ord == types.Norm(1):
			if colAxis > rowAxis {
				colAxis--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(rowAxis); err != nil {
				return
			}
			return retVal.Max(colAxis)
		case ord == types.Norm(-1):
			if colAxis > rowAxis {
				colAxis--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(rowAxis); err != nil {
				return
			}
			return retVal.Min(colAxis)
		case ord == types.Norm(0):
			err = types.NewError(types.OpError, "Norm of order 0 undefined for matrices")
			return

		case ord.IsInf(1):
			if rowAxis > colAxis {
				rowAxis--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(colAxis); err != nil {
				return
			}
			return retVal.Max(rowAxis)
		case ord.IsInf(-1):
			if rowAxis > colAxis {
				rowAxis--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(colAxis); err != nil {
				return
			}
			return retVal.Min(rowAxis)
		case ord.IsUnordered() || ord.IsFrobenius():
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = PointwiseSquare(retVal); err != nil {
				return
			}
			if retVal, err = retVal.Sum(axes...); err != nil {
				return
			}
			return Sqrt(retVal)
		case ord.IsNuclear():
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAxis, colAxis); err != nil {
				return
			}
			return retVal.Sum(len(t.Shape()) - 1)
		case ord == types.Norm(0):
			err = types.NewError(types.OpError, "Norm order 0 undefined for matrices")
			return
		default:
			return nil, notyetimplemented("Axes %v, ord %v", axes, ord)
		}
	default:
		err = dimMismatchError(2, len(axes))
		return
	}
	panic("Unreachable")
}
