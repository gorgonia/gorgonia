package tensor

import (
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
)

func Dot(a, b types.Tensor, opts ...types.FuncOpt) (retVal types.Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = types.DtypeMismatchErr(a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *tf64.Tensor:
		bt := b.(*tf64.Tensor)
		return tf64.Dot(at, bt, opts...)
	case *tf32.Tensor:
		bt := b.(*tf32.Tensor)
		return tf32.Dot(at, bt, opts...)
	default:
		err = types.NewError(types.NotYetImplemented, "Dot() does not handle Tensor of %T yet", a)
	}
	return
}

func MatMul(a, b types.Tensor, opts ...types.FuncOpt) (retVal types.Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = types.DtypeMismatchErr(a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *tf64.Tensor:
		bt := b.(*tf64.Tensor)
		return at.MatMul(bt, opts...)
	case *tf32.Tensor:
		bt := b.(*tf32.Tensor)
		return at.MatMul(bt, opts...)
	default:
		err = types.NewError(types.NotYetImplemented, "MatMul() does not handle Tensor of %T yet", a)
	}
	return
}

func MatVecMul(a, b types.Tensor, opts ...types.FuncOpt) (retVal types.Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = types.DtypeMismatchErr(a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *tf64.Tensor:
		bt := b.(*tf64.Tensor)
		return at.MatVecMul(bt, opts...)
	case *tf32.Tensor:
		bt := b.(*tf32.Tensor)
		return at.MatVecMul(bt, opts...)
	default:
		err = types.NewError(types.NotYetImplemented, "MatVecMul() does not handle Tensor of %T yet", a)
	}
	return
}

func Inner(a, b types.Tensor) (retVal types.Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = types.DtypeMismatchErr(a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *tf64.Tensor:
		bt := b.(*tf64.Tensor)
		return at.Inner(bt)
	case *tf32.Tensor:
		bt := b.(*tf32.Tensor)
		return at.Inner(bt)
	default:
		err = types.NewError(types.NotYetImplemented, "Inner() does not handle Tensor of %T yet", a)
	}
	return
}

func Outer(a, b types.Tensor, opts ...types.FuncOpt) (retVal types.Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = types.DtypeMismatchErr(a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *tf64.Tensor:
		bt := b.(*tf64.Tensor)
		return at.Outer(bt, opts...)
	case *tf32.Tensor:
		bt := b.(*tf32.Tensor)
		return at.Outer(bt, opts...)
	default:
		err = types.NewError(types.NotYetImplemented, "Outer() does not handle Tensor of %T yet", a)
	}
	return
}

func Sub(a, b interface{}, opts ...types.FuncOpt) (retVal types.Tensor, err error) {
	switch a.(type) {
	case *tf64.Tensor:
		return tf64.Sub(a, b, opts...)
	case *tf32.Tensor:
		return tf32.Sub(a, b, opts...)
	case *ti.Tensor:
		return ti.Sub(a, b, opts...)
	default:
		err = types.NewError(types.NotYetImplemented, "Sub not yet implemented for %T", a)
	}
	return
}
