package value

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/perf"
	"gorgonia.org/tensor"
)

// DualValue ...
type DualValue struct {
	Value
	D Value // the derivative wrt to each input
}

// SetDeriv ...
func (dv *DualValue) SetDeriv(d Value) error {
	if t, ok := d.(tensor.Tensor); ok && t.IsScalar() {
		d, _ = AnyToScalar(t.ScalarValue())
	}
	dv.D = d

	return dv.sanity()
}

// SetValue ...
func (dv *DualValue) SetValue(v Value) error {
	dv.Value = v
	return dv.sanity()
}

// Clone ...
func (dv *DualValue) Clone() (retVal interface{}, err error) {
	var v, d Value
	if v, err = CloneValue(dv.Value); err != nil {
		return nil, errors.Wrap(err, cloneFail)
	}

	if dv.D != nil {
		if d, err = CloneValue(dv.D); err != nil {
			return nil, errors.Wrap(err, cloneFail)
		}
	}

	dv2 := BorrowDV()
	dv2.Value = v
	dv2.D = d
	retVal = dv2
	return
}

// Type ...
func (dv *DualValue) Type() hm.Type { return TypeOf(dv.Value) }

// Dtype ...
func (dv *DualValue) Dtype() tensor.Dtype { return dv.Value.Dtype() }

// ValueEq ...
func (dv *DualValue) ValueEq(a Value) bool {
	switch at := a.(type) {
	case *DualValue:
		if at == dv {
			return true
		}
		veq := Eq(at.Value, dv.Value)
		deq := Eq(at.D, dv.D)
		return veq && deq
	// case Value:
	// 	return ValueEq(at, dv.Value)
	default:
		return false
	}
}

func (dv *DualValue) String() string {
	return fmt.Sprintf("%#+v", dv.Value)
}

func (dv *DualValue) sanity() error {
	// check that d and v are the same type

	dvv := typeCheckTypeOf(dv.Value)
	dvd := typeCheckTypeOf(dv.D)
	if !dvv.Eq(dvd) {
		return errors.Errorf("DualValues do not have the same types: %v and %v", dvv, dvd)
	}
	perf.ReturnType(dvv)
	perf.ReturnType(dvd)

	// TODO: check that the shapes are the same

	return nil
}

// clones the DualValue and zeroes out the ndarrays
func (dv *DualValue) clone0() (retVal *DualValue, err error) {
	var v, d Value
	if v, err = CloneValue(dv.Value); err != nil {
		return nil, errors.Wrap(err, cloneFail)
	}

	if d, err = CloneValue(dv.D); err != nil {
		return nil, errors.Wrap(err, cloneFail)
	}

	v = ZeroValue(v)
	d = ZeroValue(d)

	dv2 := BorrowDV()
	dv2.Value = v
	dv2.D = d
	retVal = dv2
	return
}
