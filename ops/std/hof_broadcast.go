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
type Broadcast struct {
	op      ops.Op
	pattern bcPat

	a, b     shapes.Shape
	retShape shapes.Shape
}

// Auto creates a Broadcast Op.
func Auto(op ops.Op, a, b ops.Operand) (ops.Op, error) {
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
	return &Broadcast{
		op:      op,
		a:       aShape,
		b:       bShape,
		pattern: pat,
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
	retVal = op.alloc(ctx2, a, b)
	err = op.do(ctx2, retVal, a, b)
	task.End()
	return
}

func (op *Broadcast) String() string { return fmt.Sprintf("¨%v", op.op) } // The ¨ symbol is the Diaeresis symbol (U+00A8), not the Combining Diaeresis (U+0308).

// SaveRetShape allows the final shape from the shape system to be cached in the op.
func (op *Broadcast) SaveRetShape(ret shapes.Shape) { op.retShape = ret }

func (op *Broadcast) alloc(ctx context.Context, a, b tensor.Tensor) (retVal values.Value) {
	broadcastOn := op.pattern.on()
	leftOperand := broadcastOn[0]
	rightOperand := broadcastOn[1]
	_, _ = leftOperand, rightOperand
	panic("NYI")
}

func (op *Broadcast) do(ctx context.Context, prealloc, a, b values.Value) (err error) {
	panic("NYI")
}

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
