package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

// Value represents a value that Gorgonia accepts. At this point it is implemented by:
//		- all scalar value types (F64, F32... etc)
// 		- *tensor.Dense
// 		- *dualValue
//
// A Value is essentially any thing that knows its own type and shape.
// Most importantly though, a Value is a pointer - and can be converted into a Memory.
// This is done for the sake of interoperability with external devices like cgo or CUDA or OpenCL.
// This also means for the most part most Values will be allocated on the heap.
// There are some performance tradeoffs made in this decision, but ultimately this is better than having to manually manage blocks of memory
type Value interface {
	Shape() tensor.Shape // Shape  returns the shape of the Value. Scalar values return ScalarShape()
	Size() int           // Size represents the number of elements in the Value. Note that in cases such as a *tensor.Dense, the underlying slice MAY have more elements than the Size() reports. This is correct.
	Data() interface{}   // Data returns the original representation of the Value
	Dtype() tensor.Dtype // Dtype returns the Dtype of the value

	Memory
	fmt.Formatter
}

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

// Arena is a representation of a pool of Memory
type Arena interface {
	Get(dev Device, size int64) (Memory, error) // Get returns a NoOpError when it cannot get a memory. Please allocate
	Put(dev Device, mem Memory, size int64)     // puts the memory back into the arena
}

// Valuer is any type that can return a Value
type Valuer interface {
	Value() Value
}

// Zeroer is a Value that can zero itself
type Zeroer interface {
	Value
	Zero()
}

// ZeroValuer is a a Value that can provide the zero-value of its type
type ZeroValuer interface {
	Value
	ZeroValue() Value
}

// Dtyper represents any type (typically a Value) that knows its own Dtype
type Dtyper interface {
	Dtype() tensor.Dtype
}

// Typer represents any type (typically a Op) that knows its own Type
type Typer interface {
	Type() hm.Type
}

// ValueEqualer represents any type that can perform a equal value check
type ValueEqualer interface {
	ValueEq(Value) bool
}

// Cloner represents any type that can clone itself.
type Cloner interface {
	Clone() interface{}
}

// CopierTo represents any type that can copy data to the destination.
type CopierTo interface {
	CopyTo(dest interface{}) error
}

// CopierFrom represents any type that can copy data from the source provided.
type CopierFrom interface {
	CopyFrom(src interface{}) error
}

// Setter is a any value that can Memset itself to the provided value
// type Setter interface {
// 	SetAll(interface{}) error
// }

// memoryQueue is a queue of last used memories. It's adapted from Rodrigo Moraes' implementation
// which has slightly more book keeping than a simple FIFO queue. There are some simple optimization points
// that can be easily won - the calculation of head and tail uses a modulo for example - it can easily
// be eliminated with extra fields in the struct. But I'll leave that to future work when it really becomes
// a bottleneck.
//
// https://gist.github.com/moraes/2141121
type memoryQueue struct {
	data    []Memory
	memsize uint

	size  int
	head  int
	tail  int
	count int
}

func newMemoryQueue(memsize uint) *memoryQueue {
	return &memoryQueue{
		data:    make([]Memory, 32),
		memsize: memsize,

		size: 32,
	}
}

func (q *memoryQueue) add(mem Memory) {
	// resize if need be
	if q.head == q.tail && q.count > 0 {
		mems := make([]Memory, len(q.data)+q.size)
		copy(mems, q.data[q.head:])
		copy(mems[len(q.data)-q.head:], q.data[:q.head])
		q.head = 0
		q.tail = len(q.data)
		q.data = mems
	}
	q.data[q.tail] = mem
	q.tail = (q.tail + 1) % len(q.data)
	q.count++
}

func (q *memoryQueue) get() (Memory, error) {
	if q.count == 0 {
		return nil, noopError{}
	}

	mem := q.data[q.head]
	q.head = (q.head + 1) % len(q.data)
	q.count--
	return mem, nil
}

// makeValue creates a value given a type and shape. The default value is the zero value of the type.
func makeValue(t hm.Type, s tensor.Shape) (retVal Value, err error) {
	var dt tensor.Dtype
	if dt, err = dtypeOf(t); err != nil {
		return
	}

	if s.IsScalar() {
		switch dt {
		case tensor.Float64:
			return newF64(0), nil
		case tensor.Float32:
			return newF32(0), nil
		case tensor.Int:
			return newI(0), nil
		case tensor.Int64:
			return newI64(0), nil
		case tensor.Int32:
			return newI32(0), nil
		case tensor.Byte:
			return newU8(0), nil
		case tensor.Bool:
			return newB(false), nil
		}
	}

	switch tt := t.(type) {
	case TensorType:
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...)), nil
	default:
		err = errors.Errorf(nyiTypeFail, "MakeValue", tt)
		return
	}
	panic("Unreachable")
}
