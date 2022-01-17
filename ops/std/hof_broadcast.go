package stdops

import (
	"context"
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

const (
	bcAllowableAxes = 4
)

// bcPat is actually a bit array.
// It's split into 2 nibbles - the left nibble represents the left operand, the right nibble represents the right operand:
//		xxxx|xxxx
// The least significant bit of each nibble is elem 0.
// Concrete examples:
//		00000010 (0x02) = broadcast axis 1 of the right operand
//		00000001 (0x01) = broadcast axis 0 of the right operand
//		00000101 (0x09) = broadcast axis 0 AND axis 2 of the right operand
//		00010000 (0x10) = broadcast axis 0 of the left operand
//		00110000 (0x30) = broadcast axis 0 and axis 1 of the lef operand
// You get the drill.
//
// Do note that the current limitation of the `bcPat` allows only up to 4 dimensions per operand.
type bcPat byte

func (pat bcPat) bc(left bool, axis byte) bool {
	operand := axis
	if left {
		operand += bcAllowableAxes
	}
	return (byte(pat)>>operand)&byte(1) == 1
}

// on returns a slice of axes to broadcast along.
// TODO(chewxy): there are a lot of infelicities with the code involving `.on()` and the bcPat. Fix them when free.
func (pat bcPat) on() (retVal [2][]int) {
	for i := 0; i < bcAllowableAxes; i++ {
		if pat.bc(true, byte(i)) {
			retVal[0] = append(retVal[0], i)
		}
	}

	for i := 0; i < bcAllowableAxes; i++ {
		if pat.bc(false, byte(i)) {
			retVal[1] = append(retVal[1], i)
		}
	}

	return
}

// Broadcast is an Op that performs broadcasting.
// While it's modeled as a higher order function, it's not actually a fully higher order function.
//
// Consider the type signature and shape signature of a hypothetical higher-order-function `Broadcast`:
// 	(a → b → c) → a → b → c
// 	{(a → b → c) → a → b → (a|b) | IsBroadcastable(a, b)}
// While it is perfectly doable to have such a function, in real life, we're not going to be broadcasting
// that many functions. Instead, we're just going to be broadcasting a few binary functions.
//
// So instead, we parameterize the `Op`. This makes the
// constructor `Auto` quite necessary. The constructor would make any number of different kinds of
// `Broadcast` op.
type Broadcast struct {
	op      ops.Op
	pattern bcPat

	a, b     shapes.Shape
	retShape shapes.Shape
}

// Auto creates a Broadcast Op.
func Auto(op ops.Op, a, b ops.Operand) (ops.Op, error) {
	if op.Arity() != 2 {
		return nil, errors.Errorf("Broadcast only broadcasts binary operations. %v has an arity of %d", op, op.Arity())
	}
	aShape := a.Shape()
	bShape := b.Shape()

	if aShape.Dims() != bShape.Dims() {
		return nil, errors.Errorf("Unable to automatically broadcast %v. Operands should have the same dimensions. Got %v and %v instead.", op, aShape, bShape)
	}
	var pat bcPat
	for i := 0; i < aShape.Dims(); i++ {
		if aShape[i] > bShape[i] {
			if bShape[i] != 1 {
				return nil, errors.Errorf("Unable to automatically broadcast %v. Right operand %v has non-1 size in dimension %d.", op, bShape, i)
			}
			pat |= bcPat(1) << bcPat(i)
		} else if aShape[i] < bShape[i] {
			if aShape[i] != 1 {
				return nil, errors.Errorf("Unable to automatically broadcast %v. Left operand %v has non-1 size in dimension %d.", op, bShape, i)
			}
			pat |= bcPat(1) << bcPat(i+bcAllowableAxes)
		}
	}
	retShape := computeNewShape(aShape, bShape, pat.on()[0])
	retShape = computeNewShape(bShape, retShape, pat.on()[1])
	return &Broadcast{
		op:       op,
		a:        aShape,
		b:        bShape,
		pattern:  pat,
		retShape: retShape,
	}, nil

}

// Arity returns the arity of the op that is wrapped.
func (op *Broadcast) Arity() int { return op.op.Arity() }

// Type returns the type of the op that is wrapped.
func (op *Broadcast) Type() hm.Type { return op.op.Type() }

// ShapeExpr returns the shape expression of the broadcasted function.
func (op *Broadcast) ShapeExpr() shapes.Expr { return shapes.MakeArrow(op.a, op.b, op.retShape) }

// Do executes the op.
func (op *Broadcast) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	ctx2, task := trace.NewTask(ctx, op.String())
	newA, newB, prealloc := op.alloc(ctx, a, b, true)
	err = op.do(ctx2, prealloc, newA, newB, a, b)
	task.End()
	return prealloc, err
}

func (op *Broadcast) String() string { return fmt.Sprintf("¨%v", op.op) } // The ¨ symbol is the Diaeresis symbol (U+00A8), not the Combining Diaeresis (U+0308).

// PreallocDo performs the broadcasted operation with a preallocated value for the result.
func (op *Broadcast) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	if err = gctx.Handle(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	ctx2, task := trace.NewTask(ctx, op.String())
	newA, newB, _ := op.alloc(ctx, a, b, false)
	err = op.do(ctx2, prealloc, newA, newB, a, b)
	task.End()
	return prealloc, err
}

// SaveRetShape allows the final shape from the shape system to be cached in the op.
func (op *Broadcast) SaveRetShape(ret shapes.Shape) { op.retShape = ret }

// alloc allocates new repeated-on values (for `a` and `b`) as well as the preallocates the return value
func (op *Broadcast) alloc(ctx context.Context, a, b tensor.Tensor, computePrealloc bool) (newA, newB, prealloc values.Value) {
	broadcastOn := op.pattern.on()
	leftOperand := broadcastOn[0]
	rightOperand := broadcastOn[1]
	ashp := a.Shape()
	bshp := b.Shape()
	newShape := computeNewShape(ashp, bshp, leftOperand)
	if !newShape.Eq(ashp) {
		newA = tensor.New(tensor.WithShape(newShape...), tensor.WithEngine(a.Engine()), tensor.Of(a.Dtype()))
	}

	newShape = computeNewShape(bshp, newShape, rightOperand)
	if !newShape.Eq(bshp) {
		newB = tensor.New(tensor.WithShape(newShape...), tensor.WithEngine(b.Engine()), tensor.Of(b.Dtype()))
	}

	if computePrealloc {
		prealloc = tensor.New(tensor.WithShape(newShape...), tensor.WithEngine(a.Engine()), tensor.Of(a.Dtype()))
	}
	return
}

/*
   example cases:

   a (2, 1) b (2, 3) along: {1}
   repeat along axis 0, repeats = 3

   a (1, 3) b (2, 3) along: {0}
   repeat along axis 1, repeats = 2

   a (2, 1, 3) b (2, 4, 3) along: {1}
   repeat along axis 1, repeats = 4

   a (2, 1, 1) b (2, 4, 3) along: {1, 2}
   repeat along axis 1, repeats = 4
   repeat along axis 2, repeats = 3
*/
func (op *Broadcast) do(ctx context.Context, prealloc, newA, newB, a, b values.Value) (err error) {
	if err = gctx.Handle(ctx); err != nil {
		return err
	}
	on := op.pattern.on()
	if len(on[0]) > 0 {
		if err = op.repeat(on[0], newA, a, b); err != nil {
			return errors.Wrap(err, "While doing Broadcast on left operand")
		}
	} else {
		newA = a
	}

	if len(on[1]) > 0 {
		if err = op.repeat(on[1], newB, b, a); err != nil {
			return errors.Wrap(err, "While doing Broadcast on right operand")
		}
	} else {
		newB = b
	}
	p := op.op.(ops.PreallocOp)
	_, err = p.PreallocDo(ctx, prealloc, newA, newB)
	return err
}

// repeat performs the repeat operation to create a new tensor that is amenable to the binary operation.
func (op *Broadcast) repeat(along []int, prealloc, toBeRepeated, reference values.Value) (err error) {
	shp := reference.Shape()
	for _, ax := range along {
		reps := shp[ax]
		if prealloc, err = tensor.RepeatReuse(toBeRepeated, prealloc, ax, reps); err != nil {
			return errors.Wrapf(err, "Cannot repeat along axis %d of %v", ax, toBeRepeated)
		}
	}
	return nil
}

// computeNewShape assumes `a` and `b` have the same Dims().
// computeNewShape assumes broadcastAlong has the correct values (i.e. doesn't contain Dims() not in `a` or `b`)
func computeNewShape(a, b shapes.Shape, broadcastAlong []int) shapes.Shape {
	newShape := a.Clone()
	for _, i := range broadcastAlong {
		if newShape[i] != 1 {
			err := errors.Errorf("%dth dim of `a` is not broadcastable. a: %v b %v broadcastAlong %v", i, a, b, broadcastAlong)
			panic(err)
		}
		newShape[i] = b[i]
	}
	return newShape
}

// calcBCShape computes the shape to be reshaped to. e.g.
// 	a: (3,), b: (2, 3), broadcastAlong: {0}
// will compute (3,1) for `a`.
func calcBCShape(shp shapes.Shape, expectedDims int, broadcastAlong []int) (newShape shapes.Shape) {
	if shp.Dims() == expectedDims {
		newShape = shp.Clone()
	} else {
		newShape = make(shapes.Shape, expectedDims)
		for _, i := range broadcastAlong {
			newShape[i] = 1
		}
	}

	switch {
	case shp.Eq(shapes.ScalarShape()):
		for i := range newShape {
			newShape[i] = 1
		}
	case shp.Dims() == expectedDims:
	default:
		for _, s := range shp {
			// search for first non 0
			for j := range newShape {
				if newShape[j] == 0 {
					newShape[j] = s
					break
				}
			}
		}
	}
	return
}
