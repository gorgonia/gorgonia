package cuda

import (
	"fmt"
	"log"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func getDenseTensor(t tensor.Tensor) (tensor.DenseTensor, error) {
	switch tt := t.(type) {
	case tensor.DenseTensor:
		return tt, nil
	case tensor.Densor:
		return tt.Dense(), nil
	default:
		return nil, errors.Errorf("Tensor %T is not a DenseTensor", t)
	}
}

func handleFuncOpts(expShape tensor.Shape, expType tensor.Dtype, o tensor.DataOrder, strict bool, opts ...tensor.FuncOpt) (reuse tensor.DenseTensor, safe, toReuse, incr, same bool, err error) {
	fo := tensor.ParseFuncOpts(opts...)

	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDenseTensor(reuseT); err != nil {
			err = errors.Wrapf(err, "Expected a tensor.DenseTensor")
			return
		}

		if (strict || same) && reuse.Dtype() != expType {
			err = errors.Errorf(typeMismatch, expType, reuse.Dtype())
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.DataSize() != expShape.TotalSize() && !expShape.IsScalar() {
			log.Printf("REUSE CHECK reuse shape %v, expected Shape %v", reuse.Shape(), expShape)
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch - reuse.len() %v, expShape.TotalSize() %v", reuse.DataSize(), expShape.TotalSize())
			return
		}

		if !incr && reuse != nil {
			// reuse.setDataOrder(o)
			// err = reuse.reshape(expShape...)
		}

	}
	return
}

func binaryCheck(a, b tensor.Tensor) (err error) {
	at := a.Dtype()
	bt := b.Dtype()

	switch at {
	case tensor.Float32, tensor.Float64:
	default:
		return errors.Errorf("Unsupported Dtype for a: %v", at)
	}

	switch bt {
	case tensor.Float32, tensor.Float64:
	default:
		return errors.Errorf("Unsupported Dtype for b: %v", bt)
	}

	if at.Kind() != bt.Kind() {
		return errors.Errorf(typeMismatch, at, bt)
	}
	if !a.Shape().Eq(b.Shape()) {
		log.Printf("BINARY CHECK %v %v", a.Shape(), b.Shape())
		return errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
	}

	if a.RequiresIterator() {
		return errors.New("unsupported operation: a requires an iterator")
	}

	if b.RequiresIterator() {
		return errors.New("unsupported operation: b requires an iterator")
	}
	return nil
}

func unaryCheck(a tensor.Tensor) error {
	at := a.Dtype()
	switch at {
	case tensor.Float32, tensor.Float64:
	default:
		return errors.Errorf("Unsupported Dtype for a: %v", at)
	}

	if a.RequiresIterator() {
		return errors.New("unsupported operation: a requires an iterator")
	}
	return nil
}

func logicalSize(s tensor.Shape) int {
	if s.IsScalar() {
		return 1
	}
	return s.TotalSize()
}

func constructName2(a, b tensor.Tensor, fn string) (name string) {
	dt := a.Dtype()
	as := a.Shape()
	bs := b.Shape()
	switch {
	case as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.%s_ss_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	case as.IsScalar() && !bs.IsScalar():
		name = fmt.Sprintf("%v.%s_sv_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	case !as.IsScalar() && bs.IsScalar():
		name = fmt.Sprintf("%v.%s_vs_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	default:
		name = fmt.Sprintf("%v.%s_vv_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	}
	return
}

func constructName1(a tensor.Tensor, leftTensor bool, fn string) (name string) {
	dt := a.Dtype()
	if leftTensor {
		name = fmt.Sprintf("%v.%s_vs_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.%s_sv_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	}
	return
}
