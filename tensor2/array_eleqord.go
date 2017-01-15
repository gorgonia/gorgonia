package tensor

import "github.com/pkg/errors"

/*
GENERATED FILE. DO NOT EDIT
*/

/* ElEq */

func (a f64s) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64s, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a f32s) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32s, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a ints) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(ints, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a i64s) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64s, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a i32s) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32s, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a u8s) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8s, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a bs) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Boolser
	var ok bool
	if compat, ok = other.(Boolser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bools()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(bs, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = true
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

/* Gt */

func (a f64s) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64s, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a f32s) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32s, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a ints) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(ints, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a i64s) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64s, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a i32s) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32s, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a u8s) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8s, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

/* Gte */

func (a f64s) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64s, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a f32s) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32s, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a ints) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(ints, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a i64s) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64s, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a i32s) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32s, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a u8s) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8s, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

/* Lt */

func (a f64s) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64s, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a f32s) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32s, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a ints) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(ints, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a i64s) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64s, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a i32s) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32s, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a u8s) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8s, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

/* Lte */

func (a f64s) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64s, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a f32s) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32s, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a ints) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(ints, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a i64s) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64s, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a i32s) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32s, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a u8s) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8s, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}
