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
	AllocAccessible() bool                    // AllocAccessible returns true if the engine return Go-accessible memory pointers?
	Alloc(size int64) (Memory, error)         // Alloc allocates memory
	Free(mem Memory, size int64) error        // Free rees memory
	Memset(mem Memory, val interface{}) error // Memset - duh
	Memclr(mem Memory)                        // Memclr - duh
	Memcpy(dst, src Memory) error             // Memcpy - duh
	Accessible(mem Memory) (Memory, error)    // Accessible returns Go-accesible memory pointers, or errors, if it cannot be done
	WorksWith(order DataOrder) bool           // WorksWith returns true if the data order can be directly worked with
}

type standardEngine interface {
	Engine

	Adder
	Suber
	Muler
	Diver
	Power
	Moder
	FMAer
	MatMuler
	MatVecMuler
	OuterProder
	Dotter
	SVDer
	Lter
	Lteer
	Gter
	Gteer
	ElEqer

	// Anything that returns interface{} cannot be added here because they will likely have additional
	// optimized versions of the functions for types.
	// For example: Tracer and InnerProder both have optimized interfaces for Float32 and Float64 which returns those types specifically.
}

type arrayMaker interface {
	makeArray(arr *array, t Dtype, size int)
}

/* Data Agnostic Execution Engine Methods */

// Transposer is any engine that can perform an unsafe transpose of a tensor.
type Transposer interface {
	Transpose(t Tensor, expStrides []int) error
}

// Concater is any enegine that can concatenate multiple Tensors together
type Concater interface {
	Concat(t Tensor, axis int, others ...Tensor) (Tensor, error)
}

// Stacker is any engine that can stack multiple Tenosrs along an axis
type Stacker interface {
	Stack(t Tensor, axis int, others ...Tensor) (Tensor, error)
}

// DenseStacker is any engine that can stack DenseTensors along an axis. This is a specialization of Stacker.
type DenseStacker interface {
	StackDense(t DenseTensor, axis int, others ...DenseTensor) (retVal DenseTensor, err error)
}

// Repeater is any engine that can repeat values along the given axis.
type Repeater interface {
	Repeat(t Tensor, axis int, repeats ...int) (Tensor, error)
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

// Suber is any engine that can perform elementwise subtraction.
type Suber interface {
	// Sub performs a - b
	Sub(a, b Tensor, opts ...FuncOpt) (Tensor, error)

	// SubScalar subtracts a scalar from/to the tensor. leftTensor indicates if the tensor is the left operand.
	// Whether or not the input tensor is clobbered is left to the implementation
	SubScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Muler is any engine that can perform elementwise multiplication.
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

// Tracer is any engine that can return the trace (aka the sum of the diagonal elements).
type Tracer interface {
	Trace(a Tensor) (interface{}, error)
}

// FMAer is any engine that can perform fused multiply add functions: A * X + Y. Also known as Axpy.
type FMAer interface {
	FMA(a, x, y Tensor) (Tensor, error)
	FMAScalar(a Tensor, x interface{}, y Tensor) (Tensor, error)
}

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

// InnerProderF32 is an optimization for float32 - results are returned as float32.
type InnerProderF32 interface {
	Inner(a, b Tensor) (float32, error)
}

// InnerProderF64 is an optimization for float64 - results are returned as float64
type InnerProderF64 interface {
	Inner(a, b Tensor) (float64, error)
}

// OuterProder is any engine that can perform outer product (kronecker) multiplication
type OuterProder interface {
	Outer(a, b, preallocated Tensor) error
}

// Dotter is used to implement sparse matrices
type Dotter interface {
	Dot(a, b Tensor, opts ...FuncOpt) (Tensor, error)
}

// SVDer is any engine that can perform SVD
type SVDer interface {
	SVD(a Tensor, uv, full bool) (s, u, v Tensor, err error)
}

/* ORD INTERFACES */

// Lter is any engine that can perform the Lt operation.
type Lter interface {
	Lt(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	LtScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Lteer is any engine that can perform the Lte operation.
type Lteer interface {
	Lte(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	LteScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Gter is any engine that can perform the Gt operation.
type Gter interface {
	Gt(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	GtScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

// Gteer is any engine that can perform the Gte operation.
type Gteer interface {
	Gte(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	GteScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

/* EQ INTERFACES */

// ElEqer is any engine that can perform the elementwise equality comparison operation.
type ElEqer interface {
	ElEq(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	EqScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)

	ElNe(a, b Tensor, opts ...FuncOpt) (Tensor, error)
	NeScalar(a Tensor, b interface{}, leftTensor bool, opts ...FuncOpt) (Tensor, error)
}

/* Unary Operators for Numbers */

// Mapper is any engine that can map a function onto the values of a tensor.
type Mapper interface {
	Map(fn interface{}, a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Neger is any engine that can negate the sign of the values in the tensor.d
type Neger interface {
	Neg(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Inver is any engine that can perform 1/x for each element in the Tensor.
type Inver interface {
	Inv(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Squarer is any engine that can square the values elementwise in a Tensor.
type Squarer interface {
	Square(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Cuber is any engine that can cube the values elementwise in a Tensor.
type Cuber interface {
	Cube(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Exper is any engine that can perform elementwise natural exponentiation on the values in a Tensor.
type Exper interface {
	Exp(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Tanher is any engine that can perform elementwise Tanh on the values in a Tensor.
type Tanher interface {
	Tanh(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Loger is any engine that can perform natural log on the values in a Tensor.
type Loger interface {
	Log(a Tensor, opt ...FuncOpt) (Tensor, error)
}

// Log2 is any engine that can perform base-2 logarithm on the values in a Tensor.
type Log2er interface {
	Log2(a Tensor, opt ...FuncOpt) (Tensor, error)
}

// Log10er is any engine that can perform base-10 logarithm on the values in a Tensor.
type Log10er interface {
	Log10(a Tensor, opt ...FuncOpt) (Tensor, error)
}

// Sqrter is any engine that can perform square root on the values in a Tensor.
type Sqrter interface {
	Sqrt(a Tensor, opt ...FuncOpt) (Tensor, error)
}

// Cbrter is any engine that can perform cube root on the values in a Tensor.
type Cbrter interface {
	Cbrt(a Tensor, opt ...FuncOpt) (Tensor, error)
}

// InvSqrter is any engine that can perform 1/sqrt(x) on the values of a Tensor.
type InvSqrter interface {
	InvSqrt(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Signer is any engine that can perform a sign function on the values of a Tensor.
type Signer interface {
	Sign(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Abser is any engine that can perform Abs on the values of a Tensor.
type Abser interface {
	Abs(a Tensor, opts ...FuncOpt) (Tensor, error)
}

// Clamper is any engine that can clamp the values in a tensor to between min and max.
type Clamper interface {
	Clamp(a Tensor, min, max interface{}, opts ...FuncOpt) (Tensor, error)
}

/* Reduction */

// Reducer is any engine that can perform a reduction function.
type Reducer interface {
	Reduce(fn interface{}, a Tensor, axis int, defaultValue interface{}, opts ...FuncOpt) (Tensor, error)
}

// OptimizedReducer is any engine that can perform a reduction function with optimizations for the first dimension, last dimension and dimensions in between.
type OptimizedReducer interface {
	OptimizedReduce(a Tensor, axis int, firstFn, lastFn, defaultFn, defaultValue interface{}, opts ...FuncOpt) (Tensor, error)
}

// Sumer is any engine that can perform summation along an axis of a Tensor.
type Sumer interface {
	Sum(a Tensor, along ...int) (Tensor, error)
}

// Proder is any engine that can perform product along an axis of a Tensor.
type Proder interface {
	Prod(a Tensor, along ...int) (Tensor, error)
}

// Miner is any engine that can find the minimum value along an axis of a Tensor.
type Miner interface {
	Min(a Tensor, along ...int) (Tensor, error)
}

// Maxer is any engine that can find the maximum value along an axis of a Tensor.
type Maxer interface {
	Max(a Tensor, along ...int) (Tensor, error)
}

/* Arg methods */

// Argmaxer is any engine that can find the indices of the maximum values along an axis.
// By convention the returned Tensor has Dtype of Int.
type Argmaxer interface {
	Argmax(t Tensor, axis int) (Tensor, error)
}

// Argmaxer is any engine that can find the indices of the minimum values along an axis.
// By convention the returned Tensor has Dtype of Int.
type Argminer interface {
	Argmin(t Tensor, axis int) (Tensor, error)
}

/* Internal interfaces for faster shit */

type denseArgmaxer interface {
	argmaxDenseTensor(t DenseTensor, axis int) (*Dense, error)
}

type denseArgminer interface {
	argminDenseTensor(t DenseTensor, axis int) (*Dense, error)
}
