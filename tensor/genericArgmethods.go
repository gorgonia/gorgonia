package tensor

import (
	"math"

	"github.com/chewxy/math32"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* int */

func argminI(a []int) int {
	var f int
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxI(a []int) int {
	var f int
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* int8 */

func argminI8(a []int8) int {
	var f int8
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxI8(a []int8) int {
	var f int8
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* int16 */

func argminI16(a []int16) int {
	var f int16
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxI16(a []int16) int {
	var f int16
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* int32 */

func argminI32(a []int32) int {
	var f int32
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxI32(a []int32) int {
	var f int32
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* int64 */

func argminI64(a []int64) int {
	var f int64
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxI64(a []int64) int {
	var f int64
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* uint */

func argminU(a []uint) int {
	var f uint
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxU(a []uint) int {
	var f uint
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* uint8 */

func argminU8(a []uint8) int {
	var f uint8
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxU8(a []uint8) int {
	var f uint8
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* uint16 */

func argminU16(a []uint16) int {
	var f uint16
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxU16(a []uint16) int {
	var f uint16
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* uint32 */

func argminU32(a []uint32) int {
	var f uint32
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxU32(a []uint32) int {
	var f uint32
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* uint64 */

func argminU64(a []uint64) int {
	var f uint64
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxU64(a []uint64) int {
	var f uint64
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* float32 */

func argminF32(a []float32) int {
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
		if math32.IsNaN(v) || math32.IsInf(v, -1) {
			min = i
			return min
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxF32(a []float32) int {
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

		if math32.IsNaN(v) || math32.IsInf(v, 1) {
			max = i
			return max
		}
		if v > f {
			max = i
			f = v
		}
	}
	return max
}

/* float64 */

func argminF64(a []float64) int {
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
		if math.IsNaN(v) || math.IsInf(v, -1) {
			min = i
			return min
		}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

func argmaxF64(a []float64) int {
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

		if math.IsNaN(v) || math.IsInf(v, 1) {
			max = i
			return max
		}
		if v > f {
			max = i
			f = v
		}
	}
	return max
}
