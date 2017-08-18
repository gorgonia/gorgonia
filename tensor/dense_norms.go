package tensor

import (
	"math"
	"reflect"

	"github.com/chewxy/math32"
	"github.com/pkg/errors"
)

func (t *Dense) multiSVDNorm(rowAxis, colAxis int) (retVal *Dense, err error) {
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
		toStack := make([]*Dense, retVal.Shape()[0])
		for i := 0; i < retVal.Shape()[0]; i++ {
			var sliced, ithS *Dense
			if sliced, err = sliceDense(retVal, ss(i)); err != nil {
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
		err = errors.Errorf("multiSVDNorm for dimensions greater than 3")
	}

	return
}

// Norm returns the p-ordered norm of the *Dense, given the axes.
//
// This implementation is directly adapted from Numpy, which is licenced under a BSD-like licence, and can be found here: https://docs.scipy.org/doc/numpy-1.9.1/license.html
func (t *Dense) Norm(ord NormOrder, axes ...int) (retVal *Dense, err error) {
	var ret Tensor
	var ok bool
	var abs, norm0, normN interface{}
	var oneOverOrd interface{}
	switch t.t.Kind() {
	case reflect.Float64:
		abs = math.Abs
		norm0 = func(x float64) float64 {
			if x != 0 {
				return 1
			}
			return 0
		}
		normN = func(x float64) float64 {
			return math.Pow(math.Abs(x), float64(ord))
		}
		oneOverOrd = float64(1) / float64(ord)
	case reflect.Float32:
		abs = math32.Abs
		norm0 = func(x float32) float32 {
			if x != 0 {
				return 1
			}
			return 0
		}
		normN = func(x float32) float32 {
			return math32.Pow(math32.Abs(x), float32(ord))
		}
		oneOverOrd = float32(1) / float32(ord)
	default:
		err = errors.Errorf("Norms only works on float types")
		return
	}

	dims := t.Dims()

	// simple case
	if len(axes) == 0 {
		if ord.IsUnordered() || (ord.IsFrobenius() && dims == 2) || (ord == Norm(2) && dims == 1) {
			backup := t.AP
			ap := BorrowAP(1)
			defer ReturnAP(ap)

			ap.unlock()
			ap.SetShape(t.Size())
			ap.lock()

			t.AP = ap
			if ret, err = Dot(t, t); err != nil { // returns a scalar
				err = errors.Wrapf(err, opFail, "Norm-0")
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-0")
			}

			switch t.t {
			case Float64:
				retVal.SetF64(0, math.Sqrt(retVal.GetF64(0)))
			case Float32:
				retVal.SetF32(0, math32.Sqrt(retVal.GetF32(0)))
			}
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
		cloned := t.Clone().(*Dense)
		switch {
		case ord.IsUnordered() || ord == Norm(2):
			if ret, err = Square(cloned); err != nil {
				return
			}

			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "UnorderedNorm-1")
			}

			if retVal, err = retVal.Sum(axes...); err != nil {
				return
			}

			if ret, err = Sqrt(retVal); err != nil {
				return
			}

			retVal, err = getDense(ret)
			return
		case ord.IsInf(1):
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "InfNorm-1")
			}
			return retVal.Max(axes...)
		case ord.IsInf(-1):
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "-InfNorm-1")
			}
			return retVal.Min(axes...)
		case ord == Norm(0):
			if ret, err = cloned.Apply(norm0); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-0")
			}
			return retVal.Sum(axes...)
		case ord == Norm(1):
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-1")
			}
			return retVal.Sum(axes...)
		default:
			if ret, err = cloned.Apply(normN); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-N")
			}

			if retVal, err = retVal.Sum(axes...); err != nil {
				return
			}
			retVal, err = retVal.PowScalar(oneOverOrd, true)
			return
		}
	case 2:
		rowAxis := axes[0]
		colAxis := axes[1]

		// checks
		if rowAxis < 0 {
			err = errors.Errorf("Row Axis %d is < 0", rowAxis)
			return
		}
		if colAxis < 0 {
			err = errors.Errorf("Col Axis %d is < 0", colAxis)
			return
		}

		if rowAxis == colAxis {
			err = errors.Errorf("Duplicate axes found. Row Axis: %d, Col Axis %d", rowAxis, colAxis)
			return
		}

		cloned := t.Clone().(*Dense)
		switch {
		case ord == Norm(2):
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAxis, colAxis); err != nil {
				return
			}
			dims := retVal.Dims()
			return retVal.Max(dims - 1)
		case ord == Norm(-2):
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAxis, colAxis); err != nil {
				return
			}
			dims := retVal.Dims()
			return retVal.Min(dims - 1)
		case ord == Norm(1):
			if colAxis > rowAxis {
				colAxis--
			}
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-1, axis = 2")
			}
			if retVal, err = retVal.Sum(rowAxis); err != nil {
				return
			}
			return retVal.Max(colAxis)
		case ord == Norm(-1):
			if colAxis > rowAxis {
				colAxis--
			}
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-(-1), axis = 2")
			}
			if retVal, err = retVal.Sum(rowAxis); err != nil {
				return
			}
			return retVal.Min(colAxis)
		case ord == Norm(0):
			err = errors.Errorf("Norm of order 0 undefined for matrices")
			return

		case ord.IsInf(1):
			if rowAxis > colAxis {
				rowAxis--
			}
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "InfNorm, axis = 2")
			}
			if retVal, err = retVal.Sum(colAxis); err != nil {
				return
			}
			return retVal.Max(rowAxis)
		case ord.IsInf(-1):
			if rowAxis > colAxis {
				rowAxis--
			}
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "-InfNorm, axis = 2")
			}
			if retVal, err = retVal.Sum(colAxis); err != nil {
				return
			}
			return retVal.Min(rowAxis)
		case ord.IsUnordered() || ord.IsFrobenius():
			if ret, err = cloned.Apply(abs); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Frobenius Norm, axis = 2")
			}
			if ret, err = Square(retVal); err != nil {
				return
			}
			if retVal, ok = ret.(*Dense); !ok {
				return nil, errors.Errorf(opFail, "Norm-0")
			}
			if retVal, err = retVal.Sum(axes...); err != nil {
				return
			}
			if ret, err = Sqrt(retVal); err != nil {
				return
			}
			return getDense(ret)
		case ord.IsNuclear():
			// svd norm
			if retVal, err = t.multiSVDNorm(rowAxis, colAxis); err != nil {
				return
			}
			return retVal.Sum(len(t.Shape()) - 1)
		case ord == Norm(0):
			err = errors.Errorf("Norm order 0 undefined for matrices")
			return
		default:
			return nil, errors.Errorf("Not yet implemented: Norm for Axes %v, ord %v", axes, ord)
		}
	default:
		err = errors.Errorf(dimMismatch, 2, len(axes))
		return
	}
	panic("Unreachable")
}
