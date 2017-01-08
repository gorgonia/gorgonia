package tensor

import "github.com/chewxy/gorgonia/tensor/types"

// a ConsOpt is a tensor construction option
type ConsOpt func(Tensor)

// WithBacking is a construction option for a Tensor
// Use it as such:
//		backing := []float64{1,2,3,4}
// 		t := New(WithBacking(backing))
// It can be used with other construction options like WithShape
func WithBacking(a interface{}) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			tt.data = arrayFromInterface(a)

			// if the type is not known
			if tt.t == nil {
				var err error
				tt.t, err = typeOf(tt.data)
				if err != nil {
					panic(err)
				}
			}
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
			throw := types.BorrowInts(len(dims))
			copy(throw, dims)
			tt.setShape(throw...)

			// special case for scalars
			if len(throw) == 0 {
				tt.data = makeArray(tt.t, 1)
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
		switch tt := t.(type) {
		case *Dense:
			if err := tt.data.Set(0, s); err != nil {
				panic(err)
			}
		default:
			panic("Unsupported Tensor Type")
		}
	}
	return f
}
