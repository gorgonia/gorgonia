package tensorf32

import (
	"math/rand"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/math32"
	"github.com/pkg/errors"
)

func RandomFloat32(size int) []float32 {
	r := make([]float32, size)
	for i := range r {
		r[i] = float32(rand.NormFloat64())
	}
	return r
}

// RangeFloat is inclusive of Start AND End
func RangeFloat32(start, end int) []float32 {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a float range that is negative in size")
	}

	r := make([]float32, size)
	for i, v := 0, float32(start); i < size; i++ {
		r[i] = v

		if incr {
			v++
		} else {
			v--
		}
	}
	return r
}

func reduce(f func(a, b float32) float32, def float32, l ...float32) (retVal float32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func zeroAll(a []float32) {
	for i := range a {
		a[i] = 0
	}
}

func boolsToFloat32s(a []bool) []float32 {
	retVal := make([]float32, len(a))
	for i, v := range a {
		if v {
			retVal[i] = float32(1)
		} else {
			retVal[i] = float32(0)
		}
	}
	return retVal
}

func argmax(a []float32) int {
	var f float32
	var max int
	var set bool
	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		// TODO: Maybe error instead of this?
		if math32.IsNaN(v) || math32.IsInf(v, 1) {
			max = i
			f = v
			break
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

func argmin(a []float32) int {
	var f float32
	var min int
	var set bool
	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}

		// TODO: Maybe error instead of this?
		if math32.IsNaN(v) || math32.IsInf(v, -1) {
			min = i
			f = v
			break
		}

		if v < f {
			min = i
			f = v
		}
	}
	return min
}

// reuseCheck checks a reuse tensor, and reshapes it to be the correct one
func reuseCheck(reuse *Tensor, as *Tensor) (err error) {
	if len(reuse.data) != as.Size() {
		err = types.NewError(types.ShapeMismatch, "Reused Tensor does not have expected shape %v. Got %v instead", as.Shape(), reuse.Shape())
		return
	}
	return reuseCheckShape(reuse, as.Shape())

}

func reuseCheckShape(reuse *Tensor, s types.Shape) (err error) {
	throw := types.BorrowInts(len(s))
	copy(throw, s)

	if err = reuse.reshape(throw...); err != nil {
		err = errors.Wrapf(err, reuseReshapeErr, s, reuse.DataSize())
		return
	}

	// clean up any funny things that may be in the reuse
	if reuse.old != nil {
		types.ReturnAP(reuse.old)
		reuse.old = nil
	}

	if reuse.transposeWith != nil {
		types.ReturnInts(reuse.transposeWith)
	}

	if reuse.viewOf != nil {
		reuse.viewOf = nil
	}
	return nil
}
