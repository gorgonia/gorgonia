package tensorf64

import (
	"math"

	"github.com/chewxy/gorgonia/tensor/types"
)

func (t *Tensor) Norm(ord types.NormOrder, axes []int) (retVal *Tensor, err error) {
	dims := t.Opdims()

	// simple case
	if len(axes) == 0 {
		if ord.IsFrobenius() || (ord == types.Norm(2) && dims == 1) {
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
		case ord == types.NormOrder(0.0):
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
			ord++
			fn := func(x float64) float64 {
				return math.Pow(math.Abs(x), float64(ord))
			}
			if retVal, err = cloned.Apply(fn, types.UseUnsafe()); err != nil {
				return
			}
			return retVal.Sum(axes...)
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
			return nil, nil
		case ord == types.Norm(-2):
			// svd norm
			return nil, nil
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
			// case types.NuclearNorm:
			// svd norm
		}
	default:
		err = dimMismatchError(2, len(axes))
		return
	}
	panic("Unreachable")
}
