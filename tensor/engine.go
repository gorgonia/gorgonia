package tensor

import (
	"unsafe"
)

// Memory is a representation of memory of the value.
//
// The main reason for requiring both Uintptr() and Pointer() methods is because while Go currently does not have a compacting
// garbage collector, from the docs of `unsafe`:
//		Even if a uintptr holds the address of some object, the garbage collector, will not update that uintptr's value if the object moves,
//		nor will that uintptr keep the object from being reclaimed.
type Memory interface {
	Uintptr() uintptr
	MemSize() uintptr
	Pointer() unsafe.Pointer
}

// Engine is a representation of an execution engine.
// While different execution engines can have different capabilities, all execution engines must be able to make memory.
type Engine interface {
	Make(dt Dtype, size int) Memory
}

/* NUMBER INTERFACES */

type Adder interface {
	Add(a, b Tensor) (Tensor, error)
	Trans(a Tensor, b interface{}) (Tensor, error)
}

type Subber interface {
	Sub(a, b Tensor) (Tensor, error)
	TransInv(a Tensor, b interface{}) (Tensor, error)
}

type Muler interface {
	Mul(a, b Tensor) (Tensor, error)
	Scale(a Tensor, b interface{}) (Tensor, error)
}

type Diver interface {
	Div(a, b Tensor) (Tensor, error)
	ScaleInv(a Tensor, b interface{}) (Tensor, error)
}

type Power interface {
	Pow(a, b Tensor) (Tensor, error)
	PowOf(a Tensor, b interface{}) (Tensor, error)
	PowOfR(a interface{}, b tensor) (Tensor, error)
}

type MatMuler interface {
	MatMul(a, b Tensor) (Tensor, error)
}

type MatVecMuler interface {
	MatVecMul(a, b Tensor) (Tensor, error)
}

type InnerProder interface {
	Inner(a, b Tensor) (Tensor, error)
}

type OuterProder interface {
	Outer(a, b Tensor) (Tensor, error)
}

/* ORD INTERFACES */

type Lter interface {
	Lt(a, b Tensor) (Tensor, error)
	LtTS(a Tensor, b interface{}) (Tensor, error)
	LtST(a interface{}, b Tensor) (Tensor, error)
}

type Lteer interface {
	Lte(a, b Tensor) (Tensor, error)
	LteTS(a Tensor, b interface{}) (Tensor, error)
	LteST(a interface{}, b Tensor) (Tensor, error)
}

type Gter interface {
	Gt(a, b Tensor) (Tensor, error)
	GtTS(a Tensor, b interface{}) (Tensor, error)
	GtST(a interface{}, b Tensor) (Tensor, error)
}

type Gteer interface {
	Gte(a, b Tensor) (Tensor, error)
	GteTS(a Tensor, b interface{}) (Tensor, error)
	GteST(a interface{}, b Tensor) (Tensor, error)
}

/* EQ INTERFACES */

type ElEqer interface {
	ElEq(a, b Tensor) (Tensor, error)
	ElEqTS(a Tensor, b interface{}) (Tensor, error)
	ElEqST(a interface{}, b Tensor) (Tensor, error)

	ElNe(a, b Tensor) (Tensor, error)
	ElNeTS(a Tensor, b interface{}) (Tensor, error)
	ElNeST(a interface{}, b Tensor) (Tensor, error)
}

/* Unary Operators for Numbers */
