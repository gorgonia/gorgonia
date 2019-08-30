package gorgonia

import (
	"sync"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
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
	ReturnType(n.t)
	tensor.ReturnInts(n.shape)

	n.t = nil
	n.shape = nil
	n.op = nil
	n.children = nil
	n.name = ""
	n.group = ""
	n.groups = nil
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
	tensor.ReturnTensor(t)
}

func returnValue(v Value) {
	if t, ok := v.(tensor.Tensor); ok {
		returnTensor(t)
	}
}

var dimSizerPool = new(sync.Map)

func borrowDimSizers(size int) []DimSizer {
	var pool *sync.Pool
	p, ok := dimSizerPool.Load(size)

	if !ok {
		s := size
		pool = &sync.Pool{
			New: func() interface{} { return make([]DimSizer, s, s) },
		}
		dimSizerPool.Store(size, pool)
	} else {
		pool = p.(*sync.Pool)
	}
	return pool.Get().([]DimSizer)
}

func returnDimSizers(ds []DimSizer) {
	p, ok := dimSizerPool.Load(cap(ds))
	if !ok {
		return
	}
	pool := p.(*sync.Pool)
	for i := range ds {
		ds[i] = nil
	}
	pool.Put(ds)
}

var tensorTypePool = &sync.Pool{
	New: func() interface{} { return new(TensorType) },
}

func borrowTensorType() *TensorType {
	return tensorTypePool.Get().(*TensorType)
}

func returnTensorType(t *TensorType) {
	switch t {
	case vecF64, vecF32:
		return
	case matF64, matF32:
		return
	case ten3F64, ten3F32:
		return
	}
	t.Of = nil
	t.Dims = 0
	tensorTypePool.Put(t)
}

// ReturnType ...
func ReturnType(t hm.Type) {
	switch tt := t.(type) {
	case *TensorType:
		returnTensorType(tt)
	case TensorType:
		// do nothing
	case tensor.Dtype:
		// do nothing
	case *hm.FunctionType:
		hm.ReturnFnType(tt)
	}
}
