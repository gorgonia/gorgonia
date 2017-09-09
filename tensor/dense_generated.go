package tensor

import "reflect"

/*
GENERATED FILE. DO NOT EDIT
*/

// Ones creates a *Dense with the provided shape and type
func Ones(dt Dtype, shape ...int) *Dense {
	d := recycledDense(dt, shape)
	switch d.t.Kind() {
	case reflect.Int:
		d.Memset(int(1))
	case reflect.Int8:
		d.Memset(int8(1))
	case reflect.Int16:
		d.Memset(int16(1))
	case reflect.Int32:
		d.Memset(int32(1))
	case reflect.Int64:
		d.Memset(int64(1))
	case reflect.Uint:
		d.Memset(uint(1))
	case reflect.Uint8:
		d.Memset(uint8(1))
	case reflect.Uint16:
		d.Memset(uint16(1))
	case reflect.Uint32:
		d.Memset(uint32(1))
	case reflect.Uint64:
		d.Memset(uint64(1))
	case reflect.Float32:
		d.Memset(float32(1))
	case reflect.Float64:
		d.Memset(float64(1))
	case reflect.Complex64:
		d.Memset(complex64(1))
	case reflect.Complex128:
		d.Memset(complex128(1))
	case reflect.Bool:
		d.Memset(true)
	default:
		// TODO: add a Oner interface
	}
	return d
}

// I creates the identity matrix (usually a square) matrix with 1s across the diagonals, and zeroes elsewhere, like so:
//		Matrix(4,4)
// 		⎡1  0  0  0⎤
// 		⎢0  1  0  0⎥
// 		⎢0  0  1  0⎥
// 		⎣0  0  0  1⎦
// While technically an identity matrix is a square matrix, in attempt to keep feature parity with Numpy,
// the I() function allows you to create non square matrices, as well as an index to start the diagonals.
//
// For example:
//		T = I(Float64, 4, 4, 1)
// Yields:
//		⎡0  1  0  0⎤
//		⎢0  0  1  0⎥
//		⎢0  0  0  1⎥
//		⎣0  0  0  0⎦
//
// The index k can also be a negative number:
// 		T = I(Float64, 4, 4, -1)
// Yields:
// 		⎡0  0  0  0⎤
// 		⎢1  0  0  0⎥
// 		⎢0  1  0  0⎥
// 		⎣0  0  1  0⎦
func I(dt Dtype, r, c, k int) *Dense {
	ret := New(Of(dt), WithShape(r, c))
	i := k
	if k < 0 {
		i = (-k) * c
	}

	var s *Dense
	var err error
	end := c - k
	if end > r {
		s, err = sliceDense(ret, nil)
	} else {
		s, err = sliceDense(ret, rs{0, end, 1})
	}

	if err != nil {
		panic(err)
	}
	var nexts []int
	iter := NewFlatIterator(s.AP)
	nexts, err = iter.Slice(rs{i, s.Size(), c + 1})

	switch s.t.Kind() {
	case reflect.Int:
		data := s.Ints()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Int8:
		data := s.Int8s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Int16:
		data := s.Int16s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Int32:
		data := s.Int32s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Int64:
		data := s.Int64s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Uint:
		data := s.Uints()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Uint8:
		data := s.Uint8s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Uint16:
		data := s.Uint16s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Uint32:
		data := s.Uint32s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Uint64:
		data := s.Uint64s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Float32:
		data := s.Float32s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Float64:
		data := s.Float64s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Complex64:
		data := s.Complex64s()
		for _, v := range nexts {
			data[v] = 1
		}
	case reflect.Complex128:
		data := s.Complex128s()
		for _, v := range nexts {
			data[v] = 1
		}
	}
	// TODO: create Oner interface for custom types
	return ret
}
