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
// While different execution engines can have different capabilities, all execution engines must be able to allocate and free memory
type Engine interface {
	AllocAccessible() bool                    // does the engine return Go-accessible memory pointers?
	Alloc(size int64) (Memory, error)         // Allocates memory
	Free(mem Memory, size int64) error        // Frees memory
	Memset(mem Memory, val interface{}) error // Memset
	Memclr(mem Memory)                        // Memclear
	Memcpy(dst, src Memory) error             // Memcpy
	Accessible(mem Memory) (Memory, error)    // returns Go-accesible memory pointers
}

/* NUMBER INTERFACES
All these are expected to be unsafe on the first tensor
*/

// Adder is any engine that can perform elementwise addition.
type Adder interface {
	// Add performs a + b
	Add(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	// Trans performs a + b. By convention, b hasthe same data type as a
	Trans(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
}

// Subber is any engine that can perform elementwise subtraction.
type Suber interface {
	// Sub performs a - b
	Sub(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	// TransInv performs a - b. By convention, b hasthe same data type as a
	TransInv(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
}

// Mul is any engine that can perform elementwise multiplication.
// For matrix multiplication, an engine should implement MatMul() or MatVecMul() or Inner()
type Muler interface {
	Mul(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	Scale(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
}

// Diver is any engine that can perform elementwise division.
type Diver interface {
	Div(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	ScaleInv(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
}

type Power interface {
	Pow(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	PowOf(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
	PowOfR(a interface{}, b Tensor, opts ...FuncOpt) (Tensor, error)
}

/* LINEAR ALGEBRA INTERFACES */

type MatMuler interface {
	MatMul(a, b, preallocated Tensor) error
}

type MatVecMuler interface {
	MatVecMul(a, b, preallocated Tensor) error
}

type InnerProder interface {
	Inner(a, b Tensor) (interface{}, error) // Inner always returns a scalar value
}

type OuterProder interface {
	Outer(a, b, preallocated Tensor) error
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

/* EQ INTERFACES
These return the same types
*/

type ElEqer interface {
	ElEq(a, b Tensor) (Tensor, error)
	ElEqTS(a Tensor, b interface{}) (Tensor, error)
	ElEqST(a interface{}, b Tensor) (Tensor, error)

	ElNe(a, b Tensor) (Tensor, error)
	ElNeTS(a Tensor, b interface{}) (Tensor, error)
	ElNeST(a interface{}, b Tensor) (Tensor, error)
}

/* Unary Operators for Numbers */

type Squarer interface {
	Square(a Tensor) error
}

type Exper interface {
	Exp(a Tensor) error
}

type InvSqrter interface {
	InvSqrt(a Tensor) error
}

func calcMemSize(dt Dtype, size int) int64 {
	return int64(dt.Size()) * int64(size)
}
