package gorgonia

import (
	"sync"

	"gorgonia.org/gorgonia/internal/perf"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

var nodePool = &sync.Pool{
	New: func() interface{} { return new(Node) },
}

func borrowNode() *Node { return nodePool.Get().(*Node) }

func returnNode(n *Node) {
	// if the node is being returned to the pool then it should be removed from the graph that it is linked too as well
	if n.g != nil {
		n.g.RemoveNode(n.ID())
	}

	// zero out any data in the node
	perf.ReturnType(n.t)
	tensor.ReturnInts(n.shape)

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

// ReturnNode returns a node to the pool. It does not check that the *Node has been removed from the graph. USE WITH CAUTION.
func ReturnNode(n *Node) {
	n.g = nil
	returnNode(n)
}

func returnTensor(t tensor.Tensor) {
	tensor.ReturnTensor(t)
}

func returnValue(v value.Value) {
	if t, ok := v.(tensor.Tensor); ok {
		returnTensor(t)
	}
}

var dimSizerPool = make(map[int]*sync.Pool)

func borrowDimSizers(size int) []ops.DimSizer {
	pool, ok := dimSizerPool[size]
	if !ok {
		s := size
		pool = &sync.Pool{
			New: func() interface{} { return make([]ops.DimSizer, s, s) },
		}
		dimSizerPool[size] = pool
	}
	return pool.Get().([]ops.DimSizer)
}

func returnDimSizers(ds []ops.DimSizer) {
	pool, ok := dimSizerPool[cap(ds)]
	if !ok {
		return
	}
	for i := range ds {
		ds[i] = nil
	}
	pool.Put(ds)
}
