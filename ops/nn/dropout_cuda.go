// +build cuda

package nnops

import (
	"fmt"
	"hash"
	"time"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/cu"
	cudnn "gorgonia.org/cu/dnn"
	t2cudnn "gorgonia.org/cu/dnn/interop"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type dropout struct {
	*cudnn.Dropout
	seed  uint64
	xDesc *cudnn.TensorDescriptor
}

func newDropout(x *gorgonia.Node, prob float64) (*dropout, error) {
	xDesc, err := t2cudnn.Describe(x)
	if err != nil {
		return nil, err
	}

	internal, err := cudnn.NewDropout(prob)
	if err != nil {
		return nil, err
	}
	return &dropout{
		Dropout: internal,
		xDesc:   xDesc,
		seed:    uint64(time.Now().UnixNano()),
	}, nil
}

func (op *dropout) Arity() int { return 1 }

func (op *dropout) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *dropout) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *dropout) Do(...gorgonia.Value) (gorgonia.Value, error) { panic("not implemented") }
func (op *dropout) ReturnsPtr() bool                             { return true }
func (op *dropout) CallsExtern() bool                            { return true }
func (op *dropout) OverwritesInput() int                         { return -1 }
func (op *dropout) WriteHash(h hash.Hash)                        { fmt.Fprintf(h, "Dropout %v", op.Dropout.Dropout()) }
func (op *dropout) Hashcode() uint32                             { return simpleHash(op) }
func (op *dropout) String() string                               { return fmt.Sprintf("Dropout %v", op.Dropout.Dropout()) }
func (op *dropout) DiffWRT(inputs int) []bool                    { return []bool{true, false} } // it technically should be []bool{true, false}

func (op *dropout) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	diffOp := &dropoutDiff{op}
	retVal = make(gorgonia.Nodes, 1) // retVal[1] will be nil
	retVal[0], err = gorgonia.ApplyOp(diffOp, grad)
	return
}

func (op *dropout) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) error {
	panic("not implemented")
}

func (op *dropout) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	x := inputs[0]
	machine := extern.(gorgonia.CUDAMachine)
	machine.Engines()[int(dev)].DoWork()
	ctx := machine.CUDNNContexts()[int(dev)]

	var s cudnn.Memory
	var memsize uintptr
	if memsize, err = op.RequiredStateSize(ctx); err != nil {
		return nil, errors.Wrap(err, "Unable to get required state size for Dropout")
	}
	if !op.IsReady() {
		var x cu.DevicePtr
		if x, err = machine.Contexts()[int(dev)].MemAlloc(int64(memsize)); err != nil {
			return nil, errors.Wrapf(err, "Unable to allocate %v bytes of memory of scratch space for Dropout", memsize)
		}
		s = tmpWrapper(x)
		if err = op.Use(ctx, s, memsize, op.seed); err != nil {
			return nil, errors.Wrapf(err, "Unable to set dropout to use context %v", ctx)
		}
	} else {
		s = op.States()
	}

	err = ctx.DropoutForward(op.Dropout, op.xDesc, x.(cudnn.Memory), op.xDesc, prealloc.(cudnn.Memory), s, memsize)
	return prealloc, err
}

type dropoutDiff struct {
	*dropout
}

func (op *dropoutDiff) Arity() int { return 1 }

func (op *dropoutDiff) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *dropoutDiff) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (op *dropoutDiff) Do(...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented")
}

func (op *dropoutDiff) ReturnsPtr() bool { return true }

func (op *dropoutDiff) CallsExtern() bool { return true }

func (op *dropoutDiff) OverwritesInput() int { return 0 }

func (op *dropoutDiff) WriteHash(h hash.Hash) { fmt.Fprintf(h, "DropoutDiff %v", op.Dropout.Dropout()) }

func (op *dropoutDiff) Hashcode() uint32 { return simpleHash(op) }

func (op *dropoutDiff) String() string { return fmt.Sprintf("DropoutDiff %v", op.Dropout.Dropout()) }

func (op *dropoutDiff) CUDADo(extern gorgonia.External, dev gorgonia.Device, prealloc gorgonia.Value, inputs ...gorgonia.Value) (retVal gorgonia.Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	dy := inputs[0]
	machine := extern.(gorgonia.CUDAMachine)
	machine.Engines()[int(dev)].DoWork()
	ctx := machine.CUDNNContexts()[int(dev)]

	if !op.IsReady() {
		return nil, errors.New("OP is not ready")
	}

	scratch := op.States()
	memsize, _ := op.RequiredStateSize(ctx)

	err = ctx.DropoutBackward(op.Dropout, op.xDesc, dy.(cudnn.Memory),
		op.xDesc, prealloc.(cudnn.Memory), scratch, memsize)
	return prealloc, err
}

type tmpWrapper cu.DevicePtr

func (p tmpWrapper) Uintptr() uintptr { return cu.DevicePtr(p).Uintptr() }

func (p tmpWrapper) Pointer() unsafe.Pointer { return unsafe.Pointer(cu.DevicePtr(p).Uintptr()) }

func (p tmpWrapper) IsNativelyAccessible() bool { return false }
