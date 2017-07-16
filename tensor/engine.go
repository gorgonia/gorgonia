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
	DataOrder() DataOrder                     // operations this engine has uses this data order (col or row major)
}

type arrayMaker interface {
	makeArray(t Dtype, size int) array
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

	// TransInv performs a - b. By convention, b has the same data type as a
	TransInv(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
	// TransInvR performs b - a. By convention, b has the same data type as a
	TransInvR(a interface{}, b Tensor, opts ...FuncOpt) (Tensor, error)
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

	// ScaleInv performs a / b. By convention, b has the same data type as a
	ScaleInv(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
	// ScaleInv performs b / a. By convention, b has the same data type as a
	ScaleInvR(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)
}

// Power is any engine that can perform elementwise pow()
type Power interface {
	Pow(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// PowOf performs a ^ b. By convention, b has the same data type as a
	PowOf(a Tensor, b interface{}, opts ...FuncOpt) (Tensor, error)

	// PowOfR performs b ^ a. By convention, b has the same data type as a
	PowOfR(a interface{}, b Tensor, opts ...FuncOpt) (Tensor, error)
}

/* LINEAR ALGEBRA INTERFACES */

// MatMuler is any engine that can perform matrix multiplication
type MatMuler interface {
	MatMul(a, b, preallocated Tensor) error
}

// MatVecMuler is any engine that can perform matrix vector multiplication
type MatVecMuler interface {
	MatVecMul(a, b, preallocated Tensor) error
}

// InnerProder is any engine that can perform inner product multiplication
type InnerProder interface {
	Inner(a, b Tensor) (interface{}, error) // Inner always returns a scalar value
}

// OuterProder is any engine that can perform outer product (kronecker) multiplication
type OuterProder interface {
	Outer(a, b, preallocated Tensor) error
}

// UnsafeTransposer is any engine that can perform an unsafe transpose of a tensor
type UnsafeTransposer interface {
	UnsafeTranspose(t Tensor) error
}

/* ORD INTERFACES */

type Lter interface {
	Lt(a, b Tensor, asSame bool) (Tensor, error)
	LtTS(a Tensor, b interface{}, asSame bool) (Tensor, error)
	LtST(a interface{}, b Tensor, asSame bool) (Tensor, error)
}

type Lteer interface {
	Lte(a, b Tensor, asSame bool) (Tensor, error)
	LteTS(a Tensor, b interface{}, asSame bool) (Tensor, error)
	LteST(a interface{}, b Tensor, asSame bool) (Tensor, error)
}

type Gter interface {
	Gt(a, b Tensor, asSame bool) (Tensor, error)
	GtTS(a Tensor, b interface{}, asSame bool) (Tensor, error)
	GtST(a interface{}, b Tensor, asSame bool) (Tensor, error)
}

type Gteer interface {
	Gte(a, b Tensor, asSame bool) (Tensor, error)
	GteTS(a Tensor, b interface{}, asSame bool) (Tensor, error)
	GteST(a interface{}, b Tensor, asSame bool) (Tensor, error)
}

/* EQ INTERFACES
These return the same types
*/

type ElEqer interface {
	ElEq(a, b Tensor, asSame bool) (Tensor, error)
	ElEqTS(a Tensor, b interface{}, asSame bool) (Tensor, error)
	ElEqST(a interface{}, b Tensor, asSame bool) (Tensor, error)

	ElNe(a, b Tensor, asSame bool) (Tensor, error)
	ElNeTS(a Tensor, b interface{}, asSame bool) (Tensor, error)
	ElNeST(a interface{}, b Tensor, asSame bool) (Tensor, error)
}

/* Unary Operators for Numbers */

type Squarer interface {
	Square(a Tensor, opts ...FuncOpt) error
}

type Exper interface {
	Exp(a Tensor, opts ...FuncOpt) error
}

type InvSqrter interface {
	InvSqrt(a Tensor, opts ...FuncOpt) error
}

type Inver interface {
	Inv(a Tensor, opts ...FuncOpt) error
}

type Clamper interface {
	Clamp(a Tensor, min, max interface{}, opts ...FuncOpt) (Tensor, error)
}

type Signer interface {
	Sign(a Tensor, opts ...FuncOpt) (Tensor, error)
}

type Neger interface {
	Neg(a Tensor, opts ...FuncOpt) (Tensor, error)
}

func calcMemSize(dt Dtype, size int) int64 {
	return int64(dt.Size()) * int64(size)
}
