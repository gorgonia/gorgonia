package gorgonia

import "github.com/pkg/errors"

const (
	bcAllowableAxes = 4
)

// BroadcastPattern is actually a bit array.
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
// Do note that the current limitation of the BroadcastPattern allows only up to 4 dimensions per operand.
type BroadcastPattern byte

// NewBroadcastPattern is a helper function to create broadcast patterns
func NewBroadcastPattern(leftAxes, rightAxes []byte) BroadcastPattern {
	var start byte
	for _, a := range leftAxes {
		a += bcAllowableAxes
		start |= byte(1) << a
	}
	for _, a := range rightAxes {
		start |= byte(1) << a
	}
	return BroadcastPattern(start)
}

func (bcpat BroadcastPattern) bc(left bool, axis byte) bool {
	operand := axis
	if left {
		operand += bcAllowableAxes
	}
	return (byte(bcpat)>>operand)&byte(1) == 1
}

func (bcpat BroadcastPattern) on() (retVal [2][]int) {
	for i := 0; i < bcAllowableAxes; i++ {
		if bcpat.bc(true, byte(i)) {
			retVal[0] = append(retVal[0], i)
		}
	}

	for i := 0; i < bcAllowableAxes; i++ {
		if bcpat.bc(false, byte(i)) {
			retVal[1] = append(retVal[1], i)
		}
	}

	return
}

// Broadcast apply the pattern to the input nodes
// and returns two nodes suitable for a binary operator.
// Broadcast works somewhat like Numpy's broadcast, except it's now exposed as a function.
func Broadcast(a, b *Node, pattern BroadcastPattern) (*Node, *Node, error) {
	broadcastOn := pattern.on()

	var err error
	x := a
	y := b
	xshape := x.Shape()
	yshape := y.Shape()

	if len(broadcastOn[0]) > 0 {
		children := Nodes{x}
		for _, a := range broadcastOn[0] {
			if a >= yshape.Dims() {
				return nil, nil, errors.Errorf("Attempting to broadcast a on axis %d of b. But b has shape %v", a, yshape)
			}

			var size *Node
			if size, err = SizeOf(a, y); err != nil {
				return nil, nil, errors.Wrap(err, operationError)
			}
			children = append(children, size)
		}
		rep := newRepeatOp(broadcastOn[0], children)
		if x, err = ApplyOp(rep, children...); err != nil {
			return nil, nil, errors.Wrap(err, operationError)
		}
	}

	if len(broadcastOn[1]) > 0 {
		children := Nodes{y}
		for _, a := range broadcastOn[1] {
			if a >= xshape.Dims() {
				return nil, nil, errors.Errorf("Attempting to broadcast b on axis %d of a. But a has shape %v", a, xshape)
			}

			var size *Node
			if size, err = SizeOf(a, x); err != nil {
				return nil, nil, errors.Wrap(err, operationError)
			}
			children = append(children, size)
		}
		rep := newRepeatOp(broadcastOn[1], children)
		if y, err = ApplyOp(rep, children...); err != nil {
			return nil, nil, errors.Wrap(err, operationError)
		}
	}
	return x, y, nil
}
