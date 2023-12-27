package cuda

import (
	"fmt"

	"gorgonia.org/gorgonia/internal/errors"
	gtu "gorgonia.org/gorgonia/internal/tensorutils"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// used to import gtu so goimports can know how to import gtu elsewhere in the package.
var _ = gtu.HandleFuncOpts

func binaryCheck[DT any](a, b tensor.Basic[DT]) (err error) {
	if !a.Shape().Eq(b.Shape()) {
		return errors.Errorf(errors.ShapeMismatch, b.Shape(), a.Shape())
	}

	if a.RequiresIterator() {
		return errors.New("unsupported operation: a requires an iterator")
	}

	if b.RequiresIterator() {
		return errors.New("unsupported operation: b requires an iterator")
	}
	return nil
}

func unaryCheck[DT any](a tensor.Basic[DT]) error {
	if a.RequiresIterator() {
		return errors.New("unsupported operation: a requires an iterator")
	}
	return nil
}

func logicalSize(s shapes.Shape) int {
	if s.IsScalar() {
		return 1
	}
	return s.TotalSize()
}

// constructName2 constructs the built-in CUDA kernel name for an operation.
func constructBinName2(a, b tensor.Desc, fn string) (name string) {
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

// constructName1 constructs the built-in CUDA kernel name for an operation (for those with an explicit scalar passed in).
func constructBinName1(a tensor.Desc, leftTensor bool, fn string) (name string) {
	dt := a.Dtype()
	if leftTensor {
		name = fmt.Sprintf("%v.%s_vs_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	} else {
		name = fmt.Sprintf("%v.%s_sv_f%d", elemBinOpMod, fn, int(dt.Size()*8))
	}
	return
}

// constructUnOpName constructs the built in CUDA kernel name for a unary operation.
func constructUnOpName(a tensor.Desc, fn string) (name string) {
	dt := a.Dtype()
	return fmt.Sprintf("%v.%v_f%d", elemUnaryOpMod, fn, int(dt.Size()*8))
}
