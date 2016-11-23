package gorgonia

import (
	"sync"

	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
)

var nodePool = new(sync.Pool)

func init() {
	nodePool.New = func() interface{} { return new(Node) }
	types1Pool.New = func() interface{} { return make(hm.Types, 1, 1) }
	dvpool.New = func() interface{} { return new(dualValue) }
}

func borrowNode() *Node {
	n := nodePool.Get().(*Node)
	return n
}

func returnNode(n *Node) {
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

// pool for Types with size of 1
var types1Pool = new(sync.Pool)

func borrowTypes1() hm.Types {
	return types1Pool.Get().(hm.Types)
}

func returnTypes1(ts hm.Types) {
	ts[0] = nil
	types1Pool.Put(ts)
}

// handles Returning of Values

var dvpool = new(sync.Pool)

func borrowDV() *dualValue {
	return new(dualValue)
	// return dvpool.Get().(*dualValue)
}

func returnDV(dv *dualValue) {
	if dvdT, ok := dv.d.(Tensor); ok {
		returnTensor(dvdT)
	}
	if dvvT, ok := dv.Value.(Tensor); ok {
		returnTensor(dvvT)
	}

	dv.d = nil
	dv.Value = nil
	dvpool.Put(dv)
}

func returnTensor(t types.Tensor) {
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
