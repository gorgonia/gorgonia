package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/gorgonia/ops"
)

// ExternalOp is an op that contains an external context. This allows for ops to be run without needing a VM
type ExternalOp struct {
	ops.Op
	execution.Context

	Prealloc  value.Value
	Incr      value.Value // is this a Incr? IncrDoers have higher precedence over PreallocDo
	UseUnsafe bool        // Is this an unsafe op? Lowest of all "special" Dos
}

// NewExternalOp creates a new *ExternalOp.
func NewExternalOp(op ops.Op, ctx execution.Context, prealloc value.Value) *ExternalOp {
	retVal := &ExternalOp{
		Op:        op,
		Context:   ctx,
		Prealloc:  prealloc,
		UseUnsafe: false,
	}

	return retVal
}

// DetermineDevice ...
func (op *ExternalOp) DetermineDevice(inputs Nodes, output *Node) error {
	dev := output.dataOn
	var inDev execution.Device = -2
	var allSame bool
	for _, in := range inputs {
		if in.dataOn != dev {
			allSame = false
		}

		if inDev == -2 {
			inDev = in.dataOn
			continue
		}
		if in.dataOn != inDev && in.dataOn != dev {
			return errors.Errorf("Cannot automatically determine device.")
		}
	}

	if !allSame {
		return errors.Errorf("Not all the same devices")
	}
	op.Device = dev
	return nil
}

// Do performs the op,
func (op *ExternalOp) Do(vals ...value.Value) (value.Value, error) {
	if op.Device == execution.CPU {
		switch {
		case op.Incr != nil:
			if id, ok := op.Op.(ops.IncrDoer); ok {
				if err := id.IncrDo(op.Incr, vals...); err != nil {
					if ver, ok := err.(value.Valuer); ok {
						return ver.Value(), nil
					}
					return nil, err
				}
				return op.Incr, nil
			}
		case op.Prealloc != nil:
			if pd, ok := op.Op.(ops.UsePreallocDoer); ok {
				pd.UsePreallocDo(op.Prealloc, vals...)
			}
			retVal, err := op.Op.Do(vals...)
			if err != nil {
				return retVal, err
			}
			return value.Copy(op.Prealloc, retVal)
		case op.UseUnsafe:
			if ud, ok := op.Op.(ops.UnsafeDoer); ok {
				return ud.UnsafeDo(vals...)
			}
			fallthrough
		default:
			return op.Op.Do(vals...)
		}
	}

	switch o := op.Op.(type) {
	case ops.CUDADoer:
		if op.Incr != nil {
			v, err := o.CUDADo(op.External, op.Device, op.Prealloc, vals...)
			if err != nil {
				return nil, err
			}

			add := newEBOByType(addOpType, value.TypeOf(op.Incr), value.TypeOf(v))
			addOp := NewExternalOp(add, op.Context, nil)
			addOp.UseUnsafe = true
			retVal, err := addOp.Do(op.Incr, v)
			return retVal, err
		}
		return o.CUDADo(op.External, op.Device, op.Prealloc, vals...)
	case ops.CLDoer:
	case ops.IncrDoer:
		if op.Incr != nil {
			if err := o.IncrDo(op.Incr, vals...); err != nil {
				if ver, ok := err.(value.Valuer); ok {
					return ver.Value(), nil
				}
				return nil, err
			}
			return op.Incr, nil
		}
		return op.Op.Do(vals...)
	case ops.UsePreallocDoer:
		if op.Prealloc != nil {
			return o.UsePreallocDo(op.Prealloc, vals...)
		}
		return op.Op.Do(vals...)
	case ops.UnsafeDoer:
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
