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

type batchnorm struct {
	mode              cudnn.BatchNormMode
	momentum, epsilon float64

	xDesc     *cudnn.TensorDescriptor
	bnScratch *cudnn.TensorDescriptor

	training bool
}

func (op *batchnorm) Arity() int { return 5 }

func (op *batchnorm) Type() hm.Type {
	// x, scale, bias, mean, var
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *batchnorm) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *batchnorm) Do(...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented")
}

func (op *batchnorm) ReturnsPtr() bool      { return true }
func (op *batchnorm) CallsExtern() bool     { return true }
func (op *batchnorm) OverwritesInput() int  { return -1 }
func (op *batchnorm) WriteHash(h hash.Hash) { fmt.Fprintf("BatchNorm %v %v", op.momentum, op.epsilon) }
func (op *batchnorm) Hashcode() uint32      { return simpleHash(op) }
func (op *batchnorm) String() string        { return fmt.Sprintf("BatchNorm %v %v", op.momentum, op.epsilon) }

func (op *batchnorm) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	panic("not implemented")

	machine := extern.(gorgonia.CUDAMachine)
	ctx := machine.CUDNNContext()

	alpha := 0.0
	beta := 1.0
	if op.training {
		err = ctx.BatchNormalizationForwardTraining(op.mode, alpha, beta,
			op.xDesc, x.(cudnn.Memory),
			op.xDesc, prealloc.(cudnn.Memory), // yDesc, y
			op.bnScratch,
			bnScale.(cudnn.Memory),
			bnBias.(cudnn.Memory),
			exponentialAverageFactor,
			runningMean.(cudnn.Memory),
			runningVar, (cudnn.Memory),
			op.epsilon,
			cachedMean.(cudnn.Memory),
			cachedVar.(cudnn.Memory),
		)
	} else {
		err = ctx.BatchNormalizationForwardInference(op.mode, alpha, beta,
			op.xDesc*TensorDescriptor, x.(cudnn.Memory),
			op.xDesc*TensorDescriptor, y.(cudnn.Memory),
			op.bnScratch,
			bnScale.(cudnn.Memory),
			bnBias.(cudnn.Memory),
			estimatedMean.(cudnn.Memory),
			estimatedVariance.(cudnn.Memory),
			epsilon)

	}
}

func (op *batchnorm) DiffWRT(inputs int) []bool { return []bool{true, false, false, false, false} }

func (op *batchnorm) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	panic("not implemented")
}

func (op *batchnorm) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) error {
	panic("not implemented")
}

func (op *batchnorm) SetTraining() { op.training = true }
func (op *batchnorm) SetTesting()  { op.training = false }
