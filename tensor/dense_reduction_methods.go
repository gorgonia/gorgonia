package tensor

import (
	"reflect"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/pkg/errors"
)

func (t *Dense) Sum(along ...int) (retVal *Dense, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}

	monotonic, incr1 := IsMonotonicInts(along)
	if (monotonic && incr1 && len(along) == t.Dims()) || len(along) == 0 {
		hdr := t.hdr()
		var ret interface{}
		switch t.t {
		case Int:
			ret = execution.SumI(t.hdr().Ints())
		case Int8:
			ret = execution.SumI8(t.hdr().Int8s())
		case Int16:
			ret = execution.SumI16(t.hdr().Int16s())
		case Int32:
			ret = execution.SumI32(t.hdr().Int32s())
		case Int64:
			ret = execution.SumI64(t.hdr().Int64s())
		case Uint:
			ret = execution.SumU(t.hdr().Uints())
		case Uint8:
			ret = execution.SumU8(t.hdr().Uint8s())
		case Uint16:
			ret = execution.SumU16(t.hdr().Uint16s())
		case Uint32:
			ret = execution.SumU32(t.hdr().Uint32s())
		case Uint64:
			ret = execution.SumU64(t.hdr().Uint64s())
		case Float32:
			ret = execution.SumF32(t.hdr().Float32s())
		case Float64:
			ret = execution.SumF64(t.hdr().Float64s())
		case Complex64:
			ret = execution.SumC64(t.hdr().Complex64s())
		case Complex128:
			ret = execution.SumC128(t.hdr().Complex128s())
		}
		retVal = New(FromScalar(ret))
		return
	}

	var reducer Reducer
	var ok bool
	if reducer, ok = e.(Reducer); !ok {
		return nil, errors.Errorf("Execution Engine for %v does not support Reduce", t)
	}

	retVal = t
	typ := t.t.Type
	prev := -1
	dims := len(retVal.Shape())
	for _, axis := range along {
		if prev == -1 {
			prev = axis
		}
		if axis > prev {
			axis--
		}

		if axis >= dims {
			err = errors.Errorf(dimMismatch, retVal.Dims(), axis)
			return
		}
		retVal = reducer.Reduce(retVal, axis, fn, reflect.Zero(typ).Interface())
	}
	return
}
