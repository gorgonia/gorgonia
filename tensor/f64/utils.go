package tensorf64

import (
	"math"
	"math/rand"

	"github.com/chewxy/gorgonia/tensor/types"
)

func RandomFloat64(size int) []float64 {
	r := make([]float64, size)
	for i := range r {
		r[i] = rand.NormFloat64()
	}
	return r
}

// RangeFloat is inclusive of Start AND End
func RangeFloat64(start, end int) []float64 {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a float range that is negative in size")
	}

	r := make([]float64, size)
	for i, v := 0, float64(start); i < size; i++ {
		r[i] = v

		if incr {
			v++
		} else {
			v--
		}
	}
	return r
}

func reduce(f func(a, b float64) float64, def float64, l ...float64) (retVal float64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func zeroAll(a []float64) {
	for i := range a {
		a[i] = 0
	}
}

func boolsToFloat64s(a []bool) []float64 {
	retVal := make([]float64, len(a))
	for i, v := range a {
		if v {
			retVal[i] = float64(1)
		} else {
			retVal[i] = float64(0)
		}
	}
	return retVal
}

func argmax(a []float64) int {
	var f float64
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
		if math.IsNaN(v) || math.IsInf(v, 1) {
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

func argmin(a []float64) int {
	var f float64
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
		if math.IsNaN(v) || math.IsInf(v, -1) {
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
	reuse.reshape(as.Shape()...)

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
