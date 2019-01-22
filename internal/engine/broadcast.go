package engine

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
)

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
func newBroadcastPattern(leftAxes, rightAxes []byte) BroadcastPattern {
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

const (
	first byte = iota
	second
)

// NewBroadcastOperation returns a new broadcast operation to be applied on the graph
// Warning, it modify the graph
func newBroadcastOperation(from byte, broadcastOn []int) Operation {
	return func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		// check if the graph is a weighted builder
		builder, ok := g.(graph.DirectedWeightedBuilder)
		if !ok {
			return nil, errors.Errorf("Broadcast needs to modify the graph but is not a DirectedWeightedBuilder")
		}
		_, ok = g.(graph.EdgeRemover)
		if !ok {
			return nil, errors.Errorf("Broadcast needs to modify the graph but is not an EdgeRemover")
		}
		it := getOrderedChildren(g, n)
		if it.Len() != 2 {
			return nil, errors.New("Broadcast: Unexpected number of children")
		}
		children := make([]*Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*Node)
		}
		firstArg := children[0]
		secondArg := children[1]
		sizeFrom := firstArg
		//arg2 := secondArg
		if from == second {
			//arg2 = firstArg
			sizeFrom = secondArg
		}

		for i, a := range broadcastOn {
			size := builder.NewNode().(*Node)
			builder.AddNode(size)
			builder.SetWeightedEdge(builder.NewWeightedEdge(size, sizeFrom, float64(i)))
			opSize := NewSizeOf(a)
			err := g.(*ExprGraph).ApplyOp(opSize, size)
			if err != nil {
				return nil, errors.Wrap(err, operationError)
			}

			g.(graph.EdgeRemover).RemoveEdge(n.ID(), sizeFrom.ID())
			builder.SetWeightedEdge(builder.NewWeightedEdge(n, size, float64(i+2)))
		}
		repeatChildren := getOrderedNodes(g, n)
		rep := newRepeatOp(broadcastOn, repeatChildren)
		return rep, nil
	}
}

// Broadcast works somewhat like Numpy's broadcast, except it's now exposed as a function.
func Broadcast(binOp ʘBinaryOperatorType, a, b *Node, pattern BroadcastPattern) (retVal *Node, err error) {
	broadcastOn := pattern.on()

	x := a
	y := b

	if len(broadcastOn[0]) > 0 {
		children := Nodes{x}
		for _, a := range broadcastOn[0] {
			var size *Node
			if size, err = SizeOf(a, y); err != nil {
				return nil, errors.Wrap(err, operationError)
			}
			children = append(children, size)
		}
		rep := newRepeatOp(broadcastOn[0], children)
		if x, err = ApplyOp(rep, children...); err != nil {
			return nil, errors.Wrap(err, operationError)
		}
	}

	if len(broadcastOn[1]) > 0 {
		children := Nodes{y}
		for _, a := range broadcastOn[1] {
			var size *Node
			if size, err = SizeOf(a, x); err != nil {
				return nil, errors.Wrap(err, operationError)
			}
			children = append(children, size)
		}
		rep := newRepeatOp(broadcastOn[1], children)
		if y, err = ApplyOp(rep, children...); err != nil {
			return nil, errors.Wrap(err, operationError)
		}
	}

	op := newElemBinOp(binOp, x, y)
	return ApplyOp(op, x, y)
}
