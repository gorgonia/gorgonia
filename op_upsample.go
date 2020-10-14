package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"

	"gorgonia.org/tensor"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
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
	xShape := x.Shape()
	op := newUpsampleOp(xShape, scale-1)
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
func (op *upsampleOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *upsampleOp) String() string {
	return fmt.Sprintf("Upsample{}(stride: (%d))", op.stride)
}
func (op *upsampleOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	s[2] = s[2] * (1 + op.stride)
	s[3] = s[3] * (1 + op.stride)
	return s, nil
}
func (op *upsampleOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := TensorType{Dims: 4, Of: a}
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

func (op *upsampleOp) DiffWRT(inputs int) []bool { return []bool{true} }

func (op *upsampleOp) SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	input := inputs[0]

	var op2 upsampleOp
	op2 = *op
	diff := &upsampleDiffOp{op2}

	var ret *Node
	if ret, err = ApplyOp(diff, input, output, grad); err != nil {
		return nil, err
	}
	return Nodes{ret}, nil
}

type upsampleDiffOp struct {
	upsampleOp
}

func (op *upsampleDiffOp) Arity() int { return 3 }

func (op *upsampleDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := TensorType{Dims: 4, Of: a}
	return hm.NewFnType(t, t, t, t)
}

func (op *upsampleDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *upsampleDiffOp) checkInput(inputs ...Value) (in, pooled, pooledGrad tensor.Tensor, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		err = errors.Errorf("Expected input to be a tensor")
		return
	}
	if in.Shape().Dims() != 4 {
		err = errors.Errorf("Expected input to have 4 dimensions")
		return
	}

	if pooled, ok = inputs[1].(tensor.Tensor); !ok {
		err = errors.Errorf("Expected pooled to be a tensor")
		return
	}

	if pooledGrad, ok = inputs[2].(tensor.Tensor); !ok {
		err = errors.Errorf("Expected pooledGrad to be a tensor")
		return
	}
	return
}

func (op *upsampleDiffOp) Do(inputs ...Value) (retVal Value, err error) {
	var gradIn tensor.Tensor
	in, pooled, pooledGrad, err := op.checkInput(inputs...)
	if err != nil {
		return nil, err
	}
	insh := in.Shape()
	gradIn = tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(in.Shape().Clone()...), tensor.WithEngine(in.Engine()))
	b, c, h, w := insh[0], insh[1], insh[2], insh[3]
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			for hi := 0; hi < h; hi++ {
				for wi := 0; wi < w; wi++ {
					summ := 0.
					for sh := 0; sh <= op.stride; sh++ {
						for sw := 0; sw <= op.stride; sw++ {
							val, err := pooledGrad.At(bi, ci, hi*(op.stride+1)+sh, wi*(op.stride+1)+sw)
							if err != nil {
								return nil, errors.Errorf("Error accessing input data at [%v, %v, %v, %v]", bi, ci, hi, wi)
							}
							if pooled.Dtype() == tensor.Float32 {
								summ += float64(val.(float32))
							} else if pooled.Dtype() == tensor.Float64 {
								summ += val.(float64)
							}
						}
					}
					if pooled.Dtype() == tensor.Float32 {
						gradIn.SetAt(float32(summ), bi, ci, hi, wi)
					}
					if pooled.Dtype() == tensor.Float64 {
						gradIn.SetAt(summ, bi, ci, hi, wi)
					}
				}
			}
		}
	}
	return gradIn, nil
}
