package tensor

// a ConsOpt is a tensor construction option
type ConsOpt func(Tensor)

// Of is a construction option for a Tensor.
func Of(a Dtype) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			tt.t = a
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithBacking is a construction option for a Tensor
// Use it as such:
//		backing := []float64{1,2,3,4}
// 		t := New(WithBacking(backing))
// It can be used with other construction options like WithShape
func WithBacking(x interface{}) ConsOpt {
	f := func(t Tensor) {
		if x == nil {
			return
		}
		switch tt := t.(type) {
		case *Dense:
			tt.fromSlice(x)
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithShape is a construction option for a Tensor. It creates the ndarray in the required shape
func WithShape(dims ...int) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			throw := BorrowInts(len(dims))
			copy(throw, dims)
			tt.setShape(throw...)

			// special case for scalars
			if len(throw) == 0 {
				// tt.data = makeArray(tt.t, 1)
			}
		default:
			panic("Unsupported Tensor type")
		}

	}

	return f
}

// FromScalar is a construction option for representing a scalar value as a Tensor
func FromScalar(s interface{}) ConsOpt {

	f := func(t Tensor) {
		switch t.(type) {
		case *Dense:
			// if tt.data != nil {
			// 	if err := tt.data.Set(0, s); err != nil {
			// 		panic(err)
			// 	}
			// 	return
			// }

			// switch st := s.(type) {
			// case float64:
			// 	tt.data = f64s{st}
			// 	tt.t = Float64
			// case float32:
			// 	tt.data = f32s{st}
			// 	tt.t = Float32
			// case int:
			// 	tt.data = ints{st}
			// 	tt.t = Int
			// case int64:
			// 	tt.data = i64s{st}
			// 	tt.t = Int64
			// case int32:
			// 	tt.data = i32s{st}
			// 	tt.t = Int32
			// case byte:
			// 	tt.data = u8s{st}
			// 	tt.t = Byte
			// case bool:
			// 	tt.data = bs{st}
			// 	tt.t = Bool
			// case Dtyper:
			// 	dt := st.Dtype()
			// 	tt.data = makeArray(dt, 1)
			// 	tt.data.Set(0, dt.ZeroValue())
			// default:
			// 	panic("Scalar value unsupported")
			// }
		default:
			panic("Unsupported Tensor Type")
		}
	}
	return f
}
