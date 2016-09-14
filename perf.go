package gorgonia

import (
	"sync"

	tb "github.com/chewxy/gorgonia/tensor/b"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
)

var nodePool = new(sync.Pool)

func init() {
	nodePool.New = func() interface{} { return new(Node) }
	fntypePool.New = func() interface{} { return new(functionType) }
	types1Pool.New = func() interface{} { return make(Types, 1, 1) }
	typeVarPool.New = func() interface{} { return new(typeVariable) }
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

var fntypePool = new(sync.Pool)

func borrowFnType() *functionType {
	return fntypePool.Get().(*functionType)
}

func returnFnType(t *functionType) {
	typeSysLogf("returning FnType")
	enterLoggingContext()
	defer leaveLoggingContext()
	t.n = nil

	switch t0t := t.ts[0].(type) {
	case *functionType:
		returnFnType(t0t)
	case *typeVariable:
		typeSysLogf("Going to return t0t %p", t0t)
		returnTypeVar(t0t)
	}

	switch t1t := t.ts[1].(type) {
	case *functionType:
		returnFnType(t1t)
	case *typeVariable:
		typeSysLogf("Going to return t1t %p", t1t)
		returnTypeVar(t1t)
	}

	t.ts[0] = nil
	t.ts[1] = nil

	fntypePool.Put(t)
}

// pool for Types with size of 1
var types1Pool = new(sync.Pool)

func borrowTypes1() Types {
	return types1Pool.Get().(Types)
}

func returnTypes1(ts Types) {
	ts[0] = nil
	types1Pool.Put(ts)
}

// pool for typevar
var typeVarLock = new(sync.Mutex)
var usedTypeVars = make(map[*typeVariable]struct{})
var typeVarPool = new(sync.Pool)

func borrowTypeVar() *typeVariable {
	typeVarLock.Lock()
	tv := typeVarPool.Get().(*typeVariable)
	// delete(usedTypeVars, tv)
	usedTypeVars[tv] = empty
	typeSysLogf("borrowing tv %p %v", tv, tv)
	typeVarLock.Unlock()
	return tv
}

func returnTypeVar(tv *typeVariable) {
	typeSysLogf("returning tv %p %v", tv, tv)
	enterLoggingContext()
	defer leaveLoggingContext()
	typeVarLock.Lock()
	defer typeVarLock.Unlock()
	if _, ok := usedTypeVars[tv]; !ok {
		return
	}
	delete(usedTypeVars, tv)
	// usedTypeVars[tv] = empty

	switch tit := tv.instance.(type) {
	case *typeVariable:
		returnTypeVar(tit)
	case *functionType:
		returnFnType(tit)
	}
	tv.name = ""
	tv.instance = nil
	typeVarPool.Put(tv)

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

func returnTensor(t Tensor) {
	switch tt := t.Tensor.(type) {
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
