package stdops

import (
	"bytes"
	"context"
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type transposeOp struct {
	pattern shapes.Axes
}

func Transpose(pattern []int) transposeOp {
	// TODO: pattern checks
	return transposeOp{ints2axes(pattern)}
}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op transposeOp) Arity() int { return 1 }

// Type returns Tensor-4 a → ⫪[Tensor-4 a a].
func (op transposeOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	d := op.pattern.Dims()
	t := types.MakeTensorType(d, a)
	ret := types.MakeDependent(t, a)
	return hm.NewFnType(t, ret)
}

// ShapeExpr returns { a → X[b] → T X[b] a | (D X[b] = D a) },
func (op transposeOp) ShapeExpr() shapes.Expr {
	expr := shapes.Arrow{
		shapes.Var('a'),
		shapes.Arrow{
			op.pattern,
			shapes.TransposeOf{
				op.pattern,
				shapes.Var('a'),
			},
		},
	}
	st := shapes.SubjectTo{
		shapes.Eq,
		shapes.UnaryOp{shapes.Dims, op.pattern},
		shapes.UnaryOp{shapes.Dims, shapes.Var('a')},
	}
	return shapes.Compound{Expr: expr, SubjectTo: st}
}

// Do executes the op.
func (op transposeOp) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	_, task := trace.NewTask(ctx, op.String())
	pattern := tensor.BorrowInts(op.pattern.Dims())
	copy(pattern, op.pattern.AsInts())
	retVal, err = tensor.Transpose(a, pattern...)
	task.End()
	return retVal, err
}

// String returns Aᵀ{...} where `...` is the transposition pattern.
func (op transposeOp) String() string {
	var buf bytes.Buffer
	buf.WriteString("Aᵀ{")
	for i, ax := range op.pattern {
		fmt.Fprintf(&buf, "%d", ax)
		if i < len(op.pattern)-1 {
			buf.WriteString(", ")
		}
	}

	buf.WriteString("}")
	return buf.String()
}

/* DIFFERENTIATION */

// diffAxes computes the backwards pass transposition pattern.
func (op transposeOp) diffAxes() shapes.Axes {
	newPattern := make(shapes.Axes, len(op.pattern))
	for i, p := range op.pattern {
		newPattern[p] = shapes.Axis(i)
	}
	return newPattern
}

/* transposeOp implements symdiff.Op */

// DiffWRT returns []bool{true}.
func (op transposeOp) DiffWRT(i int) []bool { return []bool{true} }

// SymDiff performs the symbolic differentiation of `transposeOp`.
func (op transposeOp) SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	newPattern := op.diffAxes()
	op2 := transposeOp{pattern: newPattern}
	retVal = make([]*exprgraph.Node, 1)
	if retVal[0], err = apply(g, op2, grN(inputs[0]), grad); err == nil {
		setGroup(g, encoding.GradientCluster, retVal...)
	}
	return
}

/* transposeOp implements ADOp */

// DoDiff allows transposeOp to be automatically differentiated.
func (op transposeOp) DoDiff(ctx context.Context, inputs []Tensor, output Tensor) (err error) {
	newPattern := op.diffAxes()

	adv := exprgraph.T2T(inputs[0]).(*dual.Dual)
	bdv := exprgraph.T2T(output).(*dual.Dual)
	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	// set up new context
	ctx2, task := trace.NewTask(ctx, op.String())

	op2 := transposeOp{pattern: newPattern}

	d, err := op2.Do(ctx2, bdvd)
	if err != nil {
		return errors.Wrap(err, "Failed to perform transposeOp.DoDiff()")
	}
	// get an addOp
	add := Add(adv, d)
	if advd, err = add.PreallocDo(ctx2, advd, advd, d); err != nil {
		return errors.Wrap(err, "Failed to perform addition of gradients in transposeOp.DoDiff()")
	}
	task.End()
	return nil
}
