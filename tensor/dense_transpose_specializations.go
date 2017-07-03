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
			t.SetB(i, saved)
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
		tmp = t.GetB(i)
		t.SetB(i, saved)
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
			t.SetI(i, saved)
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
		tmp = t.GetI(i)
		t.SetI(i, saved)
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
			t.SetI8(i, saved)
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
		tmp = t.GetI8(i)
		t.SetI8(i, saved)
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
			t.SetI16(i, saved)
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
		tmp = t.GetI16(i)
		t.SetI16(i, saved)
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
			t.SetI32(i, saved)
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
		tmp = t.GetI32(i)
		t.SetI32(i, saved)
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
			t.SetI64(i, saved)
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
		tmp = t.GetI64(i)
		t.SetI64(i, saved)
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
			t.SetU(i, saved)
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
		tmp = t.GetU(i)
		t.SetU(i, saved)
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
			t.SetU8(i, saved)
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
		tmp = t.GetU8(i)
		t.SetU8(i, saved)
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
			t.SetU16(i, saved)
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
		tmp = t.GetU16(i)
		t.SetU16(i, saved)
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
			t.SetU32(i, saved)
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
		tmp = t.GetU32(i)
		t.SetU32(i, saved)
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
			t.SetU64(i, saved)
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
		tmp = t.GetU64(i)
		t.SetU64(i, saved)
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
			t.SetF32(i, saved)
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
		tmp = t.GetF32(i)
		t.SetF32(i, saved)
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
			t.SetF64(i, saved)
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
		tmp = t.GetF64(i)
		t.SetF64(i, saved)
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
			t.SetC64(i, saved)
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
		tmp = t.GetC64(i)
		t.SetC64(i, saved)
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
			t.SetC128(i, saved)
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
		tmp = t.GetC128(i)
		t.SetC128(i, saved)
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
			t.SetStr(i, saved)
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
		tmp = t.GetStr(i)
		t.SetStr(i, saved)
		saved = tmp

		i = dest
	}
}

func (t *Dense) transpose(expStrides []int) {
	switch t.t.Kind() {
	case reflect.Bool:
		t.transposeB(expStrides)
	case reflect.Int:
		t.transposeI(expStrides)
	case reflect.Int8:
		t.transposeI8(expStrides)
	case reflect.Int16:
		t.transposeI16(expStrides)
	case reflect.Int32:
		t.transposeI32(expStrides)
	case reflect.Int64:
		t.transposeI64(expStrides)
	case reflect.Uint:
		t.transposeU(expStrides)
	case reflect.Uint8:
		t.transposeU8(expStrides)
	case reflect.Uint16:
		t.transposeU16(expStrides)
	case reflect.Uint32:
		t.transposeU32(expStrides)
	case reflect.Uint64:
		t.transposeU64(expStrides)
	case reflect.Float32:
		t.transposeF32(expStrides)
	case reflect.Float64:
		t.transposeF64(expStrides)
	case reflect.Complex64:
		t.transposeC64(expStrides)
	case reflect.Complex128:
		t.transposeC128(expStrides)
	case reflect.String:
		t.transposeStr(expStrides)
	default:
		axes := t.transposeWith
		size := t.len()
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
				t.Set(i, saved)
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
			tmp = t.Get(i)
			t.Set(i, saved)
			saved = tmp

			i = dest
		}
	}
}
