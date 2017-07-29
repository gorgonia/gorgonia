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

/* Data Agnostic Execution Engine Methods */

// Transposer is any engine that can perform an unsafe transpose of a tensor.
type Transposer interface {
	Transpose(t Tensor) error
}

type Materializer interface {
	Materialize(t Tensor) (Tensor, error)
}

/* NUMBER INTERFACES
All these are expected to be unsafe on the first tensor
*/

// Adder is any engine that can perform elementwise addition.
type Adder interface {
	// Add performs a + b
	Add(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// AddScalar adds a scalar to the tensor. leftTensor indicates if the tensor is the left operand.
	// Whether or not the input tensor is clobbered is left to the implementation
	AddScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Subber is any engine that can perform elementwise subtraction.
type Suber interface {
	// Sub performs a - b
	Sub(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// SubScalar subtracts a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
	// Whether or not the input tensor is clobbered is left to the implementation
	SubScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Mul is any engine that can perform elementwise multiplication.
// For matrix multiplication, an engine should implement MatMul() or MatVecMul() or Inner()
type Muler interface {
	// Mul performs a * b
	Mul(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// MulScalar multiplies a scalar to the tensor. leftTensor indicates if the tensor is the left operand.
	// Whether or not the input tensor is clobbered is left to the implementation
	MulScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Diver is any engine that can perform elementwise division.
type Diver interface {
	// Div performs a / b
	Div(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// DivScalar divides a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
	// Whether or not the input tensor is clobbered is left to the implementation
	DivScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Power is any engine that can perform elementwise Pow()
type Power interface {
	// Pow performs a ^ b
	Pow(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// PowScalar exponentiates a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
	// Whether or not the input tensor is clobbered is left to the implementation
	PowScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Moder is any engine that can perform elementwise Mod()
type Moder interface {
	// Mod performs a % b
	Mod(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// ModScalar performs a % b where one of the operands is scalar. leftTensor indicates if the tensor is the left operand.
	// Whether or not hte input tensor is clobbered is left to the implementation
	ModScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
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

/* ORD INTERFACES */

type Lter interface {
	Lt(a, b Tensor, asSame bool) (Tensor, error)
	LtScalar(a Tensor, b interface{}, leftTensor, asSame bool) (Tensor, error)
}

type Lteer interface {
	Lte(a, b Tensor, asSame bool) (Tensor, error)
	LteScalar(a Tensor, b interface{}, leftTensor, asSame bool) (Tensor, error)
}

type Gter interface {
	Gt(a, b Tensor, asSame bool) (Tensor, error)
	GtScalar(a Tensor, b interface{}, leftTensor, asSame bool) (Tensor, error)
}

type Gteer interface {
	Gte(a, b Tensor, asSame bool) (Tensor, error)
	GteScalar(a Tensor, b interface{}, leftTensor, asSame bool) (Tensor, error)
}

/* EQ INTERFACES */

type ElEqer interface {
	ElEq(a, b Tensor, asSame bool) (Tensor, error)
	ElEqScalar(a Tensor, b interface{}, leftTensor, asSame bool) (Tensor, error)

	ElNe(a, b Tensor, asSame bool) (Tensor, error)
	ElNeScalar(a Tensor, b interface{}, leftTensor, asSame bool) (Tensor, error)
}

/* Unary Operators for Numbers */

type Mapper interface {
	Map(a Tensor, opts ...FuncOpt) error
}

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

/* Reduction */

type Reducer interface {
	Reduce(a Tensor, axis int, fn, defaultValue interface{}, opts ...FuncOpt) error
}

type Sumer interface {
	Sum(a Tensor, along ...int) (Tensor, error)
}

type Proder interface {
	Prod(a Tensor, along ...int) (Tensor, error)
}

type Miner interface {
}

type Maxer interface {
}
