package tensor

import "reflect"

/*
GENERATED FILE. DO NOT EDIT
*/

/* bool */

func (t *Dense) transposeB(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp bool
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setB(i, saved)
			saved = false
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getB(i)
		t.setB(i, saved)
		saved = tmp

		i = dest
	}
}

/* int */

func (t *Dense) transposeI(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp int
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setI(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getI(i)
		t.setI(i, saved)
		saved = tmp

		i = dest
	}
}

/* int8 */

func (t *Dense) transposeI8(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp int8
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setI8(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getI8(i)
		t.setI8(i, saved)
		saved = tmp

		i = dest
	}
}

/* int16 */

func (t *Dense) transposeI16(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp int16
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setI16(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getI16(i)
		t.setI16(i, saved)
		saved = tmp

		i = dest
	}
}

/* int32 */

func (t *Dense) transposeI32(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp int32
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setI32(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getI32(i)
		t.setI32(i, saved)
		saved = tmp

		i = dest
	}
}

/* int64 */

func (t *Dense) transposeI64(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp int64
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setI64(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getI64(i)
		t.setI64(i, saved)
		saved = tmp

		i = dest
	}
}

/* uint */

func (t *Dense) transposeU(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp uint
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setU(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getU(i)
		t.setU(i, saved)
		saved = tmp

		i = dest
	}
}

/* uint8 */

func (t *Dense) transposeU8(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp uint8
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setU8(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getU8(i)
		t.setU8(i, saved)
		saved = tmp

		i = dest
	}
}

/* uint16 */

func (t *Dense) transposeU16(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp uint16
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setU16(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getU16(i)
		t.setU16(i, saved)
		saved = tmp

		i = dest
	}
}

/* uint32 */

func (t *Dense) transposeU32(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp uint32
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setU32(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getU32(i)
		t.setU32(i, saved)
		saved = tmp

		i = dest
	}
}

/* uint64 */

func (t *Dense) transposeU64(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp uint64
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setU64(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getU64(i)
		t.setU64(i, saved)
		saved = tmp

		i = dest
	}
}

/* float32 */

func (t *Dense) transposeF32(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp float32
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setF32(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getF32(i)
		t.setF32(i, saved)
		saved = tmp

		i = dest
	}
}

/* float64 */

func (t *Dense) transposeF64(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp float64
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setF64(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getF64(i)
		t.setF64(i, saved)
		saved = tmp

		i = dest
	}
}

/* complex64 */

func (t *Dense) transposeC64(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp complex64
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setC64(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getC64(i)
		t.setC64(i, saved)
		saved = tmp

		i = dest
	}
}

/* complex128 */

func (t *Dense) transposeC128(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp complex128
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setC128(i, saved)
			saved = 0
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getC128(i)
		t.setC128(i, saved)
		saved = tmp

		i = dest
	}
}

/* string */

func (t *Dense) transposeStr(expStrides []int) {
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp string
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.setStr(i, saved)
			saved = ""
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.getStr(i)
		t.setStr(i, saved)
		saved = tmp

		i = dest
	}
}

func (t *Dense) transpose(expStrides []int) {
	switch t.t.Kind() {
	case reflect.Bool:
		transposeB(expStrides)
	case reflect.Int:
		transposeI(expStrides)
	case reflect.Int8:
		transposeI8(expStrides)
	case reflect.Int16:
		transposeI16(expStrides)
	case reflect.Int32:
		transposeI32(expStrides)
	case reflect.Int64:
		transposeI64(expStrides)
	case reflect.Uint:
		transposeU(expStrides)
	case reflect.Uint8:
		transposeU8(expStrides)
	case reflect.Uint16:
		transposeU16(expStrides)
	case reflect.Uint32:
		transposeU32(expStrides)
	case reflect.Uint64:
		transposeU64(expStrides)
	case reflect.Float32:
		transposeF32(expStrides)
	case reflect.Float64:
		transposeF64(expStrides)
	case reflect.Complex64:
		transposeC64(expStrides)
	case reflect.Complex128:
		transposeC128(expStrides)
	case reflect.String:
		transposeStr(expStrides)
	default:
		// first we'll create a bit-map to track which elements have been moved to their correct places
		track := NewBitMap(size)
		track.Set(0)
		track.Set(size - 1) // first and last element of a transposedon't change

		// // we start our iteration at 1, because transposing 0 does noting.
		var saved, tmp interface{}
		var i int
		for i = 1; ; {
			dest := t.transposeIndex(i, axes, expStrides)

			if track.IsSet(i) && track.IsSet(dest) {
				t.set(i, saved)
				saved = reflect.Zero(t.t.Type).Interface()

				for i < size && track.IsSet(i) {
					i++
				}

				if i >= size {
					break
				}
				continue
			}

			track.Set(i)
			tmp = t.get(i)
			t.set(i, saved)
			saved = tmp

			i = dest
		}
	}
}
