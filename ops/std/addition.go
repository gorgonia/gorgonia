package stdops

import (
	"context"
	"runtime/trace"

	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// Add is a tensor-tensor addition.
type Add struct{ binop }

// String implements fmt.Stringer.
func (op Add) String() string { return "+" }

// Do performs addition.
func (op Add) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.Add(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op Add) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.Add(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op Add) SymDiff(inputs []*exprgraph.Node, output, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	panic("NYI")
}
