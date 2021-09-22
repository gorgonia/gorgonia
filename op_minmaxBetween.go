package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

/* MIN BETWEEN */

type minBetween struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (op minBetween) Arity() int { return 2 }

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (op minBetween) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

// returns the output shape as a function of the inputs
func (op minBetween) InferShape(shps ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(shps)); err != nil {
		return nil, err
	}
	a := shps[0].(tensor.Shape)
	b := shps[1].(tensor.Shape)
	if !a.Eq(b) {
		return nil, errors.Errorf("Expected both inputs to have the same shape. Got %v and %v instead", a, b)
	}
	return a.Clone(), nil
}

// Do executes the op
func (op minBetween) Do(vs ...Value) (Value, error) {
	if err := checkArity(op, len(vs)); err != nil {
		return nil, err
	}
	a := vs[0]
	b := vs[1]

	return tensor.MinBetween(a, b)
}

// ReturnsPtr returns false
func (op minBetween) ReturnsPtr() bool { return false }

// CallsExtern returns false
func (op minBetween) CallsExtern() bool { return false }

func (op minBetween) OverwritesInput() int { return -1 }

/* Other methods */
func (op minBetween) WriteHash(h hash.Hash) { fmt.Fprintf(h, op.String()) }

func (op minBetween) Hashcode() uint32 { return simpleHash(op) }

func (op minBetween) String() string { return "MinBetween" }

func (op minBetween) UsePreallocDo(prealloc Value, vs ...Value) (Value, error) {
	if err := checkArity(op, len(vs)); err != nil {
		return nil, err
	}
	a := vs[0]
	b := vs[1]

	return tensor.MinBetween(a, b, tensor.WithReuse(prealloc.(tensor.Tensor)))
}

func (op minBetween) DiffWRT(inputs int) []bool { return []bool{true, true} }
func (op minBetween) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	return minmaxSymDiff(inputs[0], inputs[1], output, grad)
}
func (op minBetween) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	return minmaxAutoDiff(ctx, inputs[0], inputs[1], output)
}

/* MAX BETWEEN */

type maxBetween struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determaxed at runtime
func (op maxBetween) Arity() int { return 2 }

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (op maxBetween) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

// returns the output shape as a function of the inputs
func (op maxBetween) InferShape(shps ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(shps)); err != nil {
		return nil, err
	}
	a := shps[0].(tensor.Shape)
	b := shps[1].(tensor.Shape)
	if !a.Eq(b) {
		return nil, errors.Errorf("Expected both inputs to have the same shape. Got %v and %v instead", a, b)
	}
	return a.Clone(), nil
}

// Do executes the op
func (op maxBetween) Do(vs ...Value) (Value, error) {
	if err := checkArity(op, len(vs)); err != nil {
		return nil, err
	}
	a := vs[0]
	b := vs[1]

	return tensor.MaxBetween(a, b)
}

// ReturnsPtr returns false
func (op maxBetween) ReturnsPtr() bool { return false }

// CallsExtern returns false
func (op maxBetween) CallsExtern() bool { return false }

func (op maxBetween) OverwritesInput() int { return -1 }

/* Other methods */
func (op maxBetween) WriteHash(h hash.Hash) { fmt.Fprintf(h, op.String()) }

func (op maxBetween) Hashcode() uint32 { return simpleHash(op) }

func (op maxBetween) String() string { return "MaxBetween" }

func (op maxBetween) UsePreallocDo(prealloc Value, vs ...Value) (Value, error) {
	if err := checkArity(op, len(vs)); err != nil {
		return nil, err
	}
	a := vs[0]
	b := vs[1]

	return tensor.MaxBetween(a, b, tensor.WithReuse(prealloc.(tensor.Tensor)))
}

func (op maxBetween) DiffWRT(inputs int) []bool { return []bool{true, true} }
func (op maxBetween) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	return minmaxSymDiff(inputs[0], inputs[1], output, grad)
}
func (op maxBetween) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	return minmaxAutoDiff(ctx, inputs[0], inputs[1], output)
}

func minmaxSymDiff(a, b *Node, out *Node, grad *Node) (Nodes, error) {
	mask, err := Eq(a, out, true)
	if err != nil {
		return nil, err
	}
	WithGroupName(gradClust)(mask)

	gradA, err := HadamardProd(grad, mask)
	if err != nil {
		return nil, err
	}
	WithGroupName(gradClust)(gradA)
	gradB, err := Sub(grad, gradA)
	if err != nil {
		return nil, err
	}
	WithGroupName(gradClust)(gradB)
	return Nodes{gradA, gradB}, nil
}

func minmaxAutoDiff(ctx ExecutionContext, a, b *Node, output *Node) (err error) {
	// dummy for now so let's keep everything as simple as possible
	adv, bdv := getDV(a, b)
	outdv := output.boundTo.(*dualValue)

	eqOp := newElemBinOp(ltOpType, a, b)
	eqOp.retSame = true
	eq := &ExternalOp{
		Op:               eqOp,
		ExecutionContext: ctx,
	}
	ctx.Device = a.Device()
	mask, err := eq.Do(adv.Value, outdv.Value)
	if err != nil {
		return errors.Wrap(err, "Unable to get mask")
	}

	dev := a.Device()

	var gradA, gradB, gradOut Value
	var extra bool

	if gradOut, extra, err = output.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, output, dev)
	}
	if extra {
		defer ctx.PutValue(dev, gradOut)
	}

	if gradA, extra, err = a.GradOnDevice(dev, ctx.External); err != nil {
		return errors.Wrapf(err, gradOnDeviceFail, a, dev)
	}
	if extra {
		defer ctx.PutValue(dev, gradA)
	}

	mul := NewHadamardProdOp(a, output, ctx)
	mul.Incr = gradA

	var d Value
	if d, err = mul.Do(gradOut, mask); err != nil {
		return errors.Wrapf(err, "IncrDo gradA failed")
	}
	adv.SetDeriv(d)

	sub := NewSubOp(b, a, ctx)
	sub.Incr = gradB
	if d, err = sub.Do(gradOut, adv.d); err != nil {
		return errors.Wrapf(err, "IncrDo gradB failed")
	}
	bdv.SetDeriv(d)
	return nil

}
