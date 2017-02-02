package tensor

import (
	"fmt"

	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
)

type tensorConsOpt func(types.Dtype) types.ConsOpt

func New(dt types.Dtype, opts ...tensorConsOpt) types.Tensor {
	consOpts := make([]types.ConsOpt, len(opts))
	for i, opt := range opts {
		consOpts[i] = opt(dt)
	}

	switch dt {
	case types.Float64:
		return tf64.NewTensor(consOpts...)
	case types.Float32:
		return tf32.NewTensor(consOpts...)
	case types.Int:
		return ti.NewTensor(consOpts...)
	case types.Int64:
	case types.Int32:
	case types.Byte:
	case types.Bool:
		return tb.NewTensor(consOpts...)
	}
	panic("Unreachable")
}

func WithShape(s ...int) tensorConsOpt {
	f := func(dt types.Dtype) types.ConsOpt {
		switch dt {
		case types.Float64:
			return tf64.WithShape(s...)
		case types.Float32:
			return tf32.WithShape(s...)
		case types.Int:
			return ti.WithShape(s...)
		case types.Bool:
			return tb.WithShape(s...)
		default:
			panic("Not Yet Implemented")
		}
	}
	return f
}

func WithBacking(any interface{}) tensorConsOpt {
	f := func(dt types.Dtype) types.ConsOpt {
		switch dt {
		case types.Float64:
			backing := any.([]float64)
			return tf64.WithBacking(backing)
		case types.Float32:
			backing := any.([]float32)
			return tf32.WithBacking(backing)
		case types.Int:
			backing := any.([]int)
			return ti.WithBacking(backing)
		case types.Bool:
			backing := any.([]bool)
			return tb.WithBacking(backing)
		default:
			panic("Not Yet Implemented")
		}
	}
	return f
}

func AsScalar(any interface{}) tensorConsOpt {
	f := func(dt types.Dtype) types.ConsOpt {
		switch dt {
		case types.Float64:
			s := any.(float64)
			return tf64.AsScalar(s)
		case types.Float32:
			s := any.(float32)
			return tf32.AsScalar(s)
		case types.Int:
			s := any.(int)
			return ti.AsScalar(s)
		case types.Bool:
			s := any.(bool)
			return tb.AsScalar(s)
		default:
			panic("Not yet implemented")
		}
	}
	return f
}

type Argmaxer interface {
	Argmax(int) (*ti.Tensor, error)
}

type Argminer interface {
	Argmin(int) (*ti.Tensor, error)
}

func Clone(t types.Tensor) types.Tensor {
	switch tt := t.(type) {
	case *tf64.Tensor:
		return tt.Clone()
	case *tf32.Tensor:
		return tt.Clone()
	case *ti.Tensor:
		return tt.Clone()
	case *tb.Tensor:
		return tt.Clone()
	default:
		panic(fmt.Sprintf("Cloning not yet implemented for %T", t))
	}
	panic("Unreachable")
}

func Copy(to, from types.Tensor) error {
	switch ft := from.(type) {
	case *tf64.Tensor:
		tt, ok := to.(*tf64.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	case *tf32.Tensor:
		tt, ok := to.(*tf32.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	case *ti.Tensor:
		tt, ok := to.(*ti.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	case *tb.Tensor:
		tt, ok := to.(*tb.Tensor)
		if !ok {
			return types.NewError(types.DtypeMismatch, "%T and %T are not of the same type", from, to)
		}

		return ft.CopyTo(tt)
	default:
		return types.NewError(types.NotYetImplemented, "Copying data not yet implemented for %T", from)
	}
	return nil
}

func Ones(dt types.Dtype, sizes ...int) types.Tensor {
	switch dt {
	case types.Float64:
		return tf64.Ones(sizes...)
	case types.Float32:
		return tf32.Ones(sizes...)
	case types.Int:
		return ti.Ones(sizes...)
	case types.Bool:
		return tb.Ones(sizes...)
	default:
		panic(fmt.Sprintf("dt of %v not handled yet", dt))
	}
	panic("unreachabale")
}

func Zeroes(dt types.Dtype, sizes ...int) types.Tensor {
	switch dt {
	case types.Float64:
		return tf64.Zeroes(sizes...)
	case types.Float32:
		return tf32.Zeroes(sizes...)
	case types.Int:
		return ti.Zeroes(sizes...)
	case types.Bool:
		return tb.Zeroes(sizes...)
	default:
		panic(fmt.Sprintf("dt of %v not handled yet", dt))
	}
	panic("unreachable")
}
