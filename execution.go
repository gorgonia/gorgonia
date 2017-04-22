package gorgonia

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
	UseUnsafe bool
}

// NewExternalOp creates a new *ExternalOp.
func NewExternalOp(op Op, ctx ExecutionContext, prealloc Value) *ExternalOp {
	return &ExternalOp{
		Op:               op,
		ExecutionContext: ctx,
		Prealloc:         prealloc,
		UseUnsafe:        false,
	}
}

// Do performs the op,
func (op *ExternalOp) Do(vals ...Value) (Value, error) {
	switch o := op.Op.(type) {
	case CUDADoer:
		if op.Prealloc == nil {
			return o.CUDADo(op.External, op.Device, vals[0], vals...)
		}
		return o.CUDADo(op.External, op.Device, op.Prealloc, vals...)
	case CLDoer:
	case UnsafeDoer:
		if op.UseUnsafe {
			return o.UnsafeDo(vals...)
		}
		return op.Do(vals...)
	case UsePreallocDoer:
		return o.UsePreallocDo(op.Prealloc, vals...)
	default:
		return o.Do(vals...)
	}
	panic("Unreachable")
}

func (op *ExternalOp) String() string {
	return op.Op.String()
}
