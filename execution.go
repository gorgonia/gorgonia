package gorgonia

import "unsafe"

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
	Get(dev Device, size int64) (Memory, error)       // Get returns a NoOpError when it cannot get a memory. Please allocate
	GetFromValue(dev Device, v Value) (Memory, error) // Gets a memory and copies the values into the memory and returns it.
	Put(dev Device, mem Memory, size int64)           // puts the memory back into the arena
	PutValue(dev Device, v Value)                     // puts the memory back into the arena

	// Transfers memory from device to device
	Transfer(toDev, fromDev Device, v Value, synchronous bool) (retVal Value, err error)
}

// External is a representation of an external device (cuda/cgo/openCL), conceptually modelled as a machine.
type External interface {
	Arena
	HasFunc(string) bool
	Signal() // signals the machine to do work
	Sync() chan struct{}
}

// ExecutionContext informs how an op should be executed
type ExecutionContext struct {
	External
	Device
}

// ExternalOp is an op that contains an external context. This allows for ops to be run without needing a VM
type ExternalOp struct {
	Op
	ExecutionContext

	Prealloc  Value
	Incr      Value // is this a Incr? IncrDoers have higher precedence over PreallocDo
	UseUnsafe bool  // Is this an unsafe op? Lowest of all "special" Dos
	UseCPU    bool  // forces uses of CPU, even if the op is CUDA/CL op
}

// NewExternalOp creates a new *ExternalOp.
func NewExternalOp(op Op, ctx ExecutionContext, prealloc Value) *ExternalOp {
	retVal := &ExternalOp{
		Op:               op,
		ExecutionContext: ctx,
		Prealloc:         prealloc,
		UseUnsafe:        false,
	}
	if retVal.Device == CPU {
		retVal.UseCPU = true
	}

	logf("NewExternalOp: %v", retVal.Device)

	return retVal
}

// Do performs the op,
func (op *ExternalOp) Do(vals ...Value) (Value, error) {
	if op.UseCPU {
		switch {
		case op.Incr != nil:
			if id, ok := op.Op.(IncrDoer); ok {
				if err := id.IncrDo(op.Incr, vals...); err != nil {
					if ver, ok := err.(Valuer); ok {
						return ver.Value(), nil
					}
					return nil, err
				}
				return op.Incr, nil
			}
		case op.Prealloc != nil:
			if pd, ok := op.Op.(UsePreallocDoer); ok {
				pd.UsePreallocDo(op.Prealloc, vals...)
			}
			retVal, err := op.Op.Do(vals...)
			if err != nil {
				return retVal, err
			}
			return Copy(op.Prealloc, retVal)
		case op.UseUnsafe:
			if ud, ok := op.Op.(UnsafeDoer); ok {
				return ud.UnsafeDo(vals...)
			}
			fallthrough
		default:
			return op.Op.Do(vals...)
		}
	}

	switch o := op.Op.(type) {
	case CUDADoer:
		if op.Incr != nil {
			v, err := o.CUDADo(op.External, op.Device, op.Prealloc, vals...)
			if err != nil {
				return nil, err
			}

			add := newEBOByType(addOpType, TypeOf(op.Incr), TypeOf(v))
			addOp := NewExternalOp(add, op.ExecutionContext, nil)
			addOp.UseUnsafe = true
			retVal, err := addOp.Do(op.Incr, v)
			return retVal, err
		}
		return o.CUDADo(op.External, op.Device, op.Prealloc, vals...)
	case CLDoer:
	case IncrDoer:
		if op.Incr != nil {
			if err := o.IncrDo(op.Incr, vals...); err != nil {
				if ver, ok := err.(Valuer); ok {
					return ver.Value(), nil
				}
				return nil, err
			}
			return op.Incr, nil
		}
		return op.Op.Do(vals...)
	case UsePreallocDoer:
		if op.Prealloc != nil {
			return o.UsePreallocDo(op.Prealloc, vals...)
		}
		return op.Op.Do(vals...)
	case UnsafeDoer:
		if op.UseUnsafe {
			return o.UnsafeDo(vals...)
		}
		return op.Op.Do(vals...)
	default:
		return o.Do(vals...)
	}

	panic("Unreachable")
}

func (op *ExternalOp) String() string {
	return op.Op.String()
}
