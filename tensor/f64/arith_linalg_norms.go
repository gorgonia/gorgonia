package tensorf64

import (
	"math"

	"github.com/chewxy/gorgonia/tensor/types"
)

func (t *Tensor) multiSVDNorm(rowAxis, colAxis int) (retVal *Tensor, err error) {
	if rowAxis > colAxis {
		rowAxis--
	}
	dims := t.Opdims()

	if retVal, err = t.RollAxis(colAxis, dims, true); err != nil {
		return
	}

	if retVal, err = retVal.RollAxis(rowAxis, dims, true); err != nil {
		return
	}

	retVal, _, _, err = retVal.SVD(false, false)
	return
}

func (t *Tensor) Norm(ord types.NormOrder, axes ...int) (retVal *Tensor, err error) {
	dims := t.Opdims()

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
		rowAx := axes[0]
		colAx := axes[1]

		if rowAx < 0 {
			// no negative indexing. error
		}
		if colAx < 0 {
			// no negative indexing. error
		}

		if rowAx == colAx {
			// duplicate axes given, error
		}

		cloned := t.Clone()
		switch {
		case ord == types.Norm(2):
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAx, colAx); err != nil {
				return
			}
			return retVal.Max(len(t.Shape()) - 1)

			// return nil, notyetimplemented("Axes %v, ord %v", axes, ord)
		case ord == types.Norm(-2):
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAx, colAx); err != nil {
				return
			}
			return retVal.Min(len(t.Shape()) - 1)

			// return nil, notyetimplemented("Axes %v, ord %v", axes, ord)
		case ord == types.Norm(1):
			if colAx > rowAx {
				colAx--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(rowAx); err != nil {
				return
			}
			return retVal.Max(colAx)
		case ord == types.Norm(-1):
			if colAx > rowAx {
				colAx--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(rowAx); err != nil {
				return
			}
			return retVal.Min(colAx)
		case ord == types.Norm(0):
			err = types.NewError(types.OpError, "Norm of order 0 undefined for matrices")
			return

		case ord.IsInf(1):
			if rowAx > colAx {
				rowAx--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(colAx); err != nil {
				return
			}
			return retVal.Max(rowAx)
		case ord.IsInf(-1):
			if rowAx > colAx {
				rowAx--
			}
			if retVal, err = cloned.Apply(math.Abs); err != nil {
				return
			}
			if retVal, err = retVal.Sum(colAx); err != nil {
				return
			}
			return retVal.Min(rowAx)
		case ord.IsFrobenius():
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
			if retVal, err = t.multiSVDNorm(rowAx, colAx); err != nil {
				return
			}
			return retVal.Sum(len(t.Shape()) - 1)
			// return nil, notyetimplemented("Axes %v, ord %v", axes, ord)
		case ord == types.Norm(0):
			// NaN because it's undefined
			retVal = NewTensor(AsScalar(math.NaN()))
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
