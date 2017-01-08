package tensor

import "github.com/pkg/errors"

func (a f64s) Memset(any interface{}) error {
	var v float64

	switch at := any.(type) {
	case float64:
		v = at
	case float32:
		v = float64(at)
	case int:
		v = float64(at)
	case int64:
		v = float64(at)
	case int32:
		v = float64(at)
	case byte:
		v = float64(at)
	default:
		return errors.Errorf("Unable to set %v of %T to a []float64", any, any)
	}

	for i := range a {
		a[i] = v
	}
	return nil
}

func (a f32s) Memset(any interface{}) error {
	var v float32

	switch at := any.(type) {
	case float64:
		v = float32(at)
	case float32:
		v = at
	case int:
		v = float32(at)
	case int64:
		v = float32(at)
	case int32:
		v = float32(at)
	case byte:
		v = float32(at)
	default:
		return errors.Errorf("Unable to set %v of %T to a []float32", any, any)
	}
	for i := range a {
		a[i] = v
	}
	return nil
}

func (a ints) Memset(any interface{}) error {
	var v int

	switch at := any.(type) {
	case float64:
		v = int(at)
	case float32:
		v = int(at)
	case int:
		v = at
	case int64:
		v = int(at)
	case int32:
		v = int(at)
	case byte:
		v = int(at)
	default:
		return errors.Errorf("Unable to set %v of %T to a []int", any, any)
	}
	for i := range a {
		a[i] = v
	}

	return nil
}

func (a i64s) Memset(any interface{}) error {
	var v int64

	switch at := any.(type) {
	case float64:
		v = int64(at)
	case float32:
		v = int64(at)
	case int:
		v = int64(at)
	case int64:
		v = at
	case int32:
		v = int64(at)
	case byte:
		v = int64(at)
	default:
		return errors.Errorf("Unable to set %v of %T to a []int64", any, any)
	}
	for i := range a {
		a[i] = v
	}
	return nil
}

func (a i32s) Memset(any interface{}) error {
	var v int32

	switch at := any.(type) {
	case float64:
		v = int32(at)
	case float32:
		v = int32(at)
	case int:
		v = int32(at)
	case int64:
		v = int32(at)
	case int32:
		v = int32(at)
	case byte:
		v = int32(at)
	default:
		return errors.Errorf("Unable to set %v of %T to a []int32", any, any)
	}
	for i := range a {
		a[i] = v
	}
	return nil
}

func (a u8s) Memset(any interface{}) error {
	var v byte

	switch at := any.(type) {
	case float64:
		if at < float64(0) {
			return errors.Errorf("%v overflows byte in []byte", any)
		}
		v = byte(at)
	case float32:
		if at < float32(0) {
			return errors.Errorf("%v overflows byte in []byte", any)
		}
		v = byte(at)
	case int:
		if at < 0 {
			return errors.Errorf("%v overflows byte in []byte", any)
		}
		v = byte(at)
	case int64:
		if at < int64(0) {
			return errors.Errorf("%v overflows byte in []byte", any)
		}
		v = byte(at)
	case int32:
		if at < int32(0) {
			return errors.Errorf("%v overflows byte in []byte", any)
		}
		v = byte(at)
	case byte:
		v = at
	default:
		return errors.Errorf("Unable to set %v of %T to a []byte", any, any)
	}

	for i := range a {
		a[i] = v
	}
	return nil
}

func (a bs) Memset(any interface{}) error {
	var v, ok bool
	if v, ok = any.(bool); !ok {
		return errors.Errorf("Unable to set %v of %T to []bool", any, any)
	}
	for i := range a {
		a[i] = v
	}
	return nil
}
