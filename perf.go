package gorgonia

import (
	"sync"

	"github.com/chewxy/gorgonia/tensor"
	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
)

var nodePool = &sync.Pool{
	New: func() interface{} { return new(Node) },
}

func borrowNode() *Node { return nodePool.Get().(*Node) }

func returnNode(n *Node) {
	// if the node is being returned to the pool then it should be removed from the graph that it is linked too as well
	if n.g != nil {
		n.g.RemoveNode(n)
	}

	// zero out any data in the node
	n.t = nil
	n.shape = nil
	n.op = nil
	n.children = nil
	n.name = ""
	n.group = ""
	n.g = nil
	n.boundTo = nil
	n.derivOf = nil
	n.deriv = nil
	n.hash = 0
	n.hashed = false
	n.inferredShape = false
	n.unchanged = false
	n.isStmt = false
	n.ofInterest = false

	nodePool.Put(n)
}

// handles Returning of Values

var dvpool = &sync.Pool{
	New: func() interface{} { return new(dualValue) },
}

func borrowDV() *dualValue { return dvpool.Get().(*dualValue) }

func returnDV(dv *dualValue) {
	returnValue(dv.d)
	returnValue(dv.Value)
	// if dvdT, ok := dv.d.(tensor.Tensor); ok {
	// 	returnTensor(dvdT)
	// }
	// if dvvT, ok := dv.Value.(tensor.Tensor); ok {
	// 	returnTensor(dvvT)
	// }

	dv.d = nil
	dv.Value = nil
	dvpool.Put(dv)
}

func returnTensor(t tensor.Tensor) {
	switch tt := t.(type) {
	case *tf64.Tensor:
		tf64.ReturnTensor(tt)
	case *tf32.Tensor:
		tf32.ReturnTensor(tt)
	case *ti.Tensor:
		ti.ReturnTensor(tt)
	case *tb.Tensor:
		tb.ReturnTensor(tt)
	default:
		return
	}
}

func returnValue(v Value) {
	if t, ok := v.(tensor.Tensor); ok {
		returnTensor(t)
	}
}
