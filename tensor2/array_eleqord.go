package tensor

import "github.com/pkg/errors"

/* ElEq */

func (a f64s) ElEq(other ElEq, same bool) (Array, error) {
	var b f64s

	switch ot := other.(type) {
	case f64s:
		b = ot
	case Float64ser:
		b = f64s(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a f32s) ElEq(other ElEq, same bool) (Array, error) {
	var b f32s

	switch ot := other.(type) {
	case f32s:
		b = ot
	case Float32ser:
		b = f32s(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a ints) ElEq(other ElEq, same bool) (Array, error) {
	var b ints

	switch ot := other.(type) {
	case ints:
		b = ot
	case Intser:
		b = ints(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a i64s) ElEq(other ElEq, same bool) (Array, error) {
	var b i64s

	switch ot := other.(type) {
	case i64s:
		b = ot
	case Int64ser:
		b = i64s(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a i32s) ElEq(other ElEq, same bool) (Array, error) {
	var b i32s

	switch ot := other.(type) {
	case i32s:
		b = ot
	case Int32ser:
		b = i32s(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a u8s) ElEq(other ElEq, same bool) (Array, error) {
	var b u8s

	switch ot := other.(type) {
	case u8s:
		b = ot
	case Byteser:
		b = u8s(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a bs) ElEq(other ElEq, same bool) (Array, error) {
	var b bs

	switch ot := other.(type) {
	case bs:
		b = ot
	case Boolser:
		b = bs(ot.Bools())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

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

func (a f64s) Gt(other ElOrd, same bool) (Array, error) {
	var b f64s

	switch ot := other.(type) {
	case f64s:
		b = ot
	case Float64ser:
		b = f64s(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

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

func (a f32s) Gt(other ElOrd, same bool) (Array, error) {
	var b f32s

	switch ot := other.(type) {
	case f32s:
		b = ot
	case Float32ser:
		b = f32s(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

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

func (a ints) Gt(other ElOrd, same bool) (Array, error) {
	var b ints

	switch ot := other.(type) {
	case ints:
		b = ot
	case Intser:
		b = ints(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

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

func (a i64s) Gt(other ElOrd, same bool) (Array, error) {
	var b i64s

	switch ot := other.(type) {
	case i64s:
		b = ot
	case Int64ser:
		b = i64s(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

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

func (a i32s) Gt(other ElOrd, same bool) (Array, error) {
	var b i32s

	switch ot := other.(type) {
	case i32s:
		b = ot
	case Int32ser:
		b = i32s(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

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

func (a u8s) Gt(other ElOrd, same bool) (Array, error) {
	var b u8s

	switch ot := other.(type) {
	case u8s:
		b = ot
	case Byteser:
		b = u8s(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

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

func (a f64s) Gte(other ElOrd, same bool) (Array, error) {
	var b f64s

	switch ot := other.(type) {
	case f64s:
		b = ot
	case Float64ser:
		b = f64s(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

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

func (a f32s) Gte(other ElOrd, same bool) (Array, error) {
	var b f32s

	switch ot := other.(type) {
	case f32s:
		b = ot
	case Float32ser:
		b = f32s(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

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

func (a ints) Gte(other ElOrd, same bool) (Array, error) {
	var b ints

	switch ot := other.(type) {
	case ints:
		b = ot
	case Intser:
		b = ints(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

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

func (a i64s) Gte(other ElOrd, same bool) (Array, error) {
	var b i64s

	switch ot := other.(type) {
	case i64s:
		b = ot
	case Int64ser:
		b = i64s(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

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

func (a i32s) Gte(other ElOrd, same bool) (Array, error) {
	var b i32s

	switch ot := other.(type) {
	case i32s:
		b = ot
	case Int32ser:
		b = i32s(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

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

func (a u8s) Gte(other ElOrd, same bool) (Array, error) {
	var b u8s

	switch ot := other.(type) {
	case u8s:
		b = ot
	case Byteser:
		b = u8s(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

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

func (a f64s) Lt(other ElOrd, same bool) (Array, error) {
	var b f64s

	switch ot := other.(type) {
	case f64s:
		b = ot
	case Float64ser:
		b = f64s(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

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

func (a f32s) Lt(other ElOrd, same bool) (Array, error) {
	var b f32s

	switch ot := other.(type) {
	case f32s:
		b = ot
	case Float32ser:
		b = f32s(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

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

func (a ints) Lt(other ElOrd, same bool) (Array, error) {
	var b ints

	switch ot := other.(type) {
	case ints:
		b = ot
	case Intser:
		b = ints(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

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

func (a i64s) Lt(other ElOrd, same bool) (Array, error) {
	var b i64s

	switch ot := other.(type) {
	case i64s:
		b = ot
	case Int64ser:
		b = i64s(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

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

func (a i32s) Lt(other ElOrd, same bool) (Array, error) {
	var b i32s

	switch ot := other.(type) {
	case i32s:
		b = ot
	case Int32ser:
		b = i32s(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

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

func (a u8s) Lt(other ElOrd, same bool) (Array, error) {
	var b u8s

	switch ot := other.(type) {
	case u8s:
		b = ot
	case Byteser:
		b = u8s(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

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

func (a f64s) Lte(other ElOrd, same bool) (Array, error) {
	var b f64s

	switch ot := other.(type) {
	case f64s:
		b = ot
	case Float64ser:
		b = f64s(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

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

func (a f32s) Lte(other ElOrd, same bool) (Array, error) {
	var b f32s

	switch ot := other.(type) {
	case f32s:
		b = ot
	case Float32ser:
		b = f32s(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

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

func (a ints) Lte(other ElOrd, same bool) (Array, error) {
	var b ints

	switch ot := other.(type) {
	case ints:
		b = ot
	case Intser:
		b = ints(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

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

func (a i64s) Lte(other ElOrd, same bool) (Array, error) {
	var b i64s

	switch ot := other.(type) {
	case i64s:
		b = ot
	case Int64ser:
		b = i64s(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

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

func (a i32s) Lte(other ElOrd, same bool) (Array, error) {
	var b i32s

	switch ot := other.(type) {
	case i32s:
		b = ot
	case Int32ser:
		b = i32s(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

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

func (a u8s) Lte(other ElOrd, same bool) (Array, error) {
	var b u8s

	switch ot := other.(type) {
	case u8s:
		b = ot
	case Byteser:
		b = u8s(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

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
