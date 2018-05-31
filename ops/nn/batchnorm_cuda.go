// +build cuda

package nnops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type BatchNormOp struct {
	mode              cudnn.BatchNormMode
	momentum, epsilon float64

	xDesc     *cudnn.TensorDescriptor
	bnScratch *cudnn.TensorDescriptor

	training bool
}

func (op *BatchNormOp) Arity() int { return 7 }

func (op *BatchNormOp) Type() hm.Type {
	// x, scale, bias, mean, var
	return hm.NewFnType(hm.TypeVariable('a'), // x
		hm.TypeVariable('a'), // scale
		hm.TypeVariable('a'), // bias
		hm.TypeVariable('a'), // running mean / expected mean
		hm.TypeVariable('a'), // running var / expected var
		hm.TypeVariable('a'), // cached mean
		hm.TypeVariable('a'), // cachedVar
		hm.TypeVariable('a')) // retVal
}

func (op *BatchNormOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *BatchNormOp) Do(...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented")
}

func (op *BatchNormOp) ReturnsPtr() bool     { return true }
func (op *BatchNormOp) CallsExtern() bool    { return true }
func (op *BatchNormOp) OverwritesInput() int { return -1 }
func (op *BatchNormOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "BatchNorm %v %v", op.momentum, op.epsilon)
}
func (op *BatchNormOp) Hashcode() uint32 { return simpleHash(op) }
func (op *BatchNormOp) String() string   { return fmt.Sprintf("BatchNorm %v %v", op.momentum, op.epsilon) }

func (op *BatchNormOp) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	panic("not implemented")

	machine := extern.(gorgonia.CUDAMachine)
	ctx := machine.CUDNNContext()

	x, bnScale, bnBias, mean, variance, cachedMean, cachedVar := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]

	alpha := 0.0
	beta := 1.0
	if op.training {
		err = ctx.BatchNormalizationForwardTraining(op.mode, alpha, beta,
			op.xDesc, x.(cudnn.Memory),
			op.xDesc, prealloc.(cudnn.Memory), // yDesc, y
			op.bnScratch,
			bnScale.(cudnn.Memory),
			bnBias.(cudnn.Memory),
			op.momentum,
			mean.(cudnn.Memory),
			variance.(cudnn.Memory),
			op.epsilon,
			cachedMean.(cudnn.Memory),
			cachedVar.(cudnn.Memory),
		)
	} else {
		err = ctx.BatchNormalizationForwardInference(op.mode, alpha, beta,
			op.xDesc, x.(cudnn.Memory),
			op.xDesc, prealloc.(cudnn.Memory),
			op.bnScratch,
			bnScale.(cudnn.Memory),
			bnBias.(cudnn.Memory),
			mean.(cudnn.Memory),     // expected mean
			variance.(cudnn.Memory), // expected variance
			op.epsilon)
	}
	return prealloc, err
}

func (op *BatchNormOp) DiffWRT(inputs int) []bool {
	return []bool{true, false, false, false, false, false, false}
}

func (op *BatchNormOp) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	panic("not implemented")
}

func (op *BatchNormOp) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) error {
	panic("not implemented")
}

func (op *BatchNormOp) SetTraining() { op.training = true }
func (op *BatchNormOp) SetTesting()  { op.training = false }
