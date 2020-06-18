package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
)

type upsampleOp struct {
	stride int
}

func newUpsampleOp(inputShape tensor.Shape, stride int) *upsampleOp {
	upsampleop := &upsampleOp{
		stride: stride,
	}
	return upsampleop
}

//Upsample2D -  simply upscaling Tensor by scale factor.
/*
	1, 2
	3, 4
	converts to
	1,1,2,2
	1,1,2,2
	3,3,4,4,
	3,3,4,4,
*/
func Upsample2D(x *Node, scale int) (*Node, error) {
	if scale < 1 {
		return nil, errors.Errorf("Upsample scale %v does not make sense", scale)
	}
	group := encoding.NewGroup("Upsample")
	xShape := x.Shape()
	op := newUpsampleOp(xShape, scale-1)
	_ = group
	retVal, err := ApplyOp(op, x)
	return retVal, err
}

func (op *upsampleOp) Arity() int {

	return 1
}
func (op *upsampleOp) ReturnsPtr() bool { return false }

func (op *upsampleOp) CallsExtern() bool { return false }

func (op *upsampleOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Upsample{}(stride: (%d))", op.stride)
}
func (op *upsampleOp) Hashcode() uint32 { return simpleHash(op) }

func (op *upsampleOp) String() string {
	return fmt.Sprintf("Upsample{}(stride: (%d))", op.stride)
}
func (op *upsampleOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}
func (op *upsampleOp) Type() hm.Type {

	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t)
}
func (op *upsampleOp) OverwritesInput() int { return -1 }

func (op *upsampleOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Expected input to have 4 dimensions")
	}
	return in, nil
}

func (op *upsampleOp) Do(inputs ...Value) (retVal Value, err error) {
	var in tensor.Tensor
	if in, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}
	inShp := in.Shape()
	b, c, h, w := inShp[0], inShp[1], inShp[2], inShp[3]

	out := tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(b, c, h*(1+op.stride), w*(1+op.stride)), tensor.WithEngine(in.Engine()))
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			for hi := 0; hi < h; hi++ {
				for wi := 0; wi < w; wi++ {
					val, err := in.At(bi, ci, hi, wi)
					if err != nil {
						return nil, errors.Errorf("Error accessing input data at [%v, %v, %v, %v]", bi, ci, hi, wi)
					}
					hout := hi * (op.stride + 1)
					wout := wi * (op.stride + 1)
					for shi := 0; shi <= op.stride; shi++ {
						for swi := 0; swi <= op.stride; swi++ {
							out.SetAt(val, bi, ci, hout+shi, wout+swi)
						}
					}
				}
			}
		}
	}

	return out, nil
}
