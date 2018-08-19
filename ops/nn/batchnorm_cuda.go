// +build cuda

package nnops

import (
	"fmt"
	"hash"
	"log"

	"github.com/chewxy/hm"
	"gorgonia.org/cu/dnn"
	t2cudnn "gorgonia.org/cu/dnn/interop"
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

func newBatchNormOp(momentum, epsilon float64) *BatchNormOp {
	return &BatchNormOp{
		mode:     cudnn.PerActivation,
		momentum: momentum,
		epsilon:  epsilon,
		training: true,
	}
}

func (op *BatchNormOp) Arity() int { return 7 }

func (op *BatchNormOp) Type() hm.Type {
	t := gorgonia.TensorType{Dims: 4, Of: hm.TypeVariable('a')}
	return hm.NewFnType(t, // x
		t, // scale
		t, // bias
		t, // running mean / expected mean
		t, // running var / expected var
		t, // cached mean
		t, // cachedVar
		t) // retVal
}

func (op *BatchNormOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *BatchNormOp) Do(...gorgonia.Value) (gorgonia.Value, error) { panic("not implemented") }
func (op *BatchNormOp) ReturnsPtr() bool                             { return true }
func (op *BatchNormOp) CallsExtern() bool                            { return true }
func (op *BatchNormOp) OverwritesInput() int                         { return -1 }
func (op *BatchNormOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "BatchNorm %v %v", op.momentum, op.epsilon)
}
func (op *BatchNormOp) Hashcode() uint32 { return simpleHash(op) }
func (op *BatchNormOp) String() string   { return fmt.Sprintf("BatchNorm %v %v", op.momentum, op.epsilon) }

func (op *BatchNormOp) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	// panic("not implemented")

	machine := extern.(gorgonia.CUDAMachine)
	ctx := machine.CUDNNContexts()[int(dev)]

	x, bnScale, bnBias, mean, variance, cachedMean, cachedVar := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
	if op.xDesc == nil {
		if op.xDesc, err = t2cudnn.Describe(x.(tensor.Tensor)); err != nil {
			return
		}
	}

	if op.bnScratch == nil {
		if op.bnScratch, err = t2cudnn.Describe(mean.(tensor.Tensor)); err != nil {
			return
		}
	}

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
			mean.(cudnn.Memory),     // runniing mean
			variance.(cudnn.Memory), // running variance
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
	return []bool{true, true, true, false, false, false, false}
}

func (op *BatchNormOp) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	x, scale, bias := inputs[0], inputs[1], inputs[2]
	cachedMean, cachedVar := inputs[5], inputs[6]
	dy := grad // rename for simplicity of reading

	// create new nodes for the diffs
	g := x.Graph()
	dt := x.Dtype()
	scaleScratch := &scratchOp{scale.Shape().Clone(), dt, scale.Name() + "Diff"}
	biasScratch := &scratchOp{bias.Shape().Clone(), dt, bias.Name() + "Diff"}
	dscale := gorgonia.NewTensor(g, dt, scale.Shape().Dims(), gorgonia.WithOp(scaleScratch))
	dbias := gorgonia.NewTensor(g, dt, bias.Shape().Dims(), gorgonia.WithOp(biasScratch))

	retVal = make(gorgonia.Nodes, 7)

	diffOp := &batchNormDiffOp{op}
	retVal[0], err = gorgonia.ApplyOp(diffOp, x, scale, dscale, dbias, dy, cachedMean, cachedVar)
	retVal[1] = dscale
	retVal[2] = dbias
	gorgonia.SetDerivOf(dscale, scale)
	gorgonia.SetDerivOf(dbias, bias)

	return retVal, err
}

func (op *BatchNormOp) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) error {
	panic("not implemented")
}

func (op *BatchNormOp) SetTraining() { op.training = true }
func (op *BatchNormOp) SetTesting()  { op.training = false }
func (op *BatchNormOp) Reset() error { return nil }

type batchNormDiffOp struct {
	*BatchNormOp
}

// Arity is the same exact function as BatchNormOp (7)

// Type is exactly the same as BatchNormOp, but the semantics are different:
// 	return hm.NewFnType(
//		t, // x
// 		t, // scale
// 		t, // dscale
// 		t, // dbias
// 		t, // dy
// 		t, // cachedMean
// 		t, // cachedVar
// 		t  // retVal
//	)

// InferShape is the same exact function as BatchNormOp

func (op *batchNormDiffOp) Do(...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented")
}

func (op *batchNormDiffOp) ReturnsPtr() bool { return true }

func (op *batchNormDiffOp) CallsExtern() bool { return true }

func (op *batchNormDiffOp) OverwritesInput() int { return -1 }

func (op *batchNormDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "BatchNormDiff %v %v", op.momentum, op.epsilon)
}

// HashCode is exactly the same as BatchNormOp

func (op *batchNormDiffOp) String() string {
	return fmt.Sprintf("BatchNormDiff %v %v", op.momentum, op.epsilon)
}

func (op *batchNormDiffOp) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	machine := extern.(gorgonia.CUDAMachine)
	e := &machine.Engines()[int(dev)]
	ctx := machine.CUDNNContexts()[int(dev)]

	x, scale, dscale, dbias, dy, cachedMean, cachedVariance := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]
	log.Printf("x %v, scale %v, dscale %v, dbias %v, dy %v, cachedMean %v, cachedVariance %v", x.Shape(), scale.Shape(), dscale.Shape(), dbias.Shape(), dy.Shape(), cachedMean.Shape(), cachedVariance.Shape())
	dscale = gorgonia.ScalarAsTensor(dscale, 4, e)
	dbias = gorgonia.ScalarAsTensor(dbias, 4, e)

	alpha := 0.0
	beta := 1.0
	err = ctx.BatchNormalizationBackward(op.mode,
		alpha, beta, // for data
		alpha, beta, // for param
		op.xDesc,
		x.(cudnn.Memory),
		op.xDesc, // dyDesc
		dy.(cudnn.Memory),
		op.xDesc, // dxDesc
		prealloc.(cudnn.Memory),
		op.bnScratch, // scratch space descriptor
		scale.(cudnn.Memory),
		dscale.(cudnn.Memory), // deriv of scale
		dbias.(cudnn.Memory),  // deriv of bias
		op.epsilon,
		cachedMean.(cudnn.Memory), cachedVariance.(cudnn.Memory))
	return prealloc, err
}
