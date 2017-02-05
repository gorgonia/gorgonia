package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
)

// Dtype is data type
type Dtype byte

const (
	Float64 Dtype = iota
	Float32
	Int
	Int64
	Int32
	Byte
	Bool

	Ptr // equivalent to interface{}. Ugh Ugh Ugh
	MAXDTYPE
)

func (t Dtype) Name() string                                  { return t.String() }
func (t Dtype) Apply(hm.Subs) hm.Substitutable                { return t }
func (t Dtype) FreeTypeVar() hm.TypeVarSet                    { return nil }
func (t Dtype) Normalize(k, v hm.TypeVarSet) (hm.Type, error) { return t, nil }
func (t Dtype) Types() hm.Types                               { return nil }
func (t Dtype) Format(state fmt.State, c rune)                { state.Write([]byte(t.String())) }
func (t Dtype) Eq(other hm.Type) bool                         { return t == other }
func (t Dtype) TensorDtype() tensor.Dtype                     { return tensor.Dtype(t) }

/*Tensor Type*/

// a TensorType is a type constructor for tensors.
//
// Think of it as  something like this:
//		data Tensor a = Tensor d a
//
// The shape of the Tensor is not part of TensorType.
// Shape checking is relegated to the dynamic part of the program run
type TensorType struct {
	Dims int // dims

	Of hm.Type
}

func fromTensorType(t TensorType, tv hm.TypeVariable) TensorType {
	retVal := newTensorType(t.Dims, tv)
	return retVal
}

func newTensorType(dims int, typ hm.Type) TensorType {
	return TensorType{
		Dims: dims,
		Of:   typ,
	}
}

func (t TensorType) Name() string { return "Tensor" }

func (t TensorType) Format(state fmt.State, c rune) {
	if state.Flag('#') {
		fmt.Fprintf(state, "Tensor-%d %#v", t.Dims, t.Of)
	} else {
		switch t.Dims {
		case 1:
			fmt.Fprintf(state, "Vector %v", t.Of)
		case 2:
			fmt.Fprintf(state, "Matrix %v", t.Of)
		default:
			fmt.Fprintf(state, "Tensor-%d %v", t.Dims, t.Of)
		}
	}
}
func (t TensorType) String() string  { return fmt.Sprintf("%v", t) }
func (t TensorType) Types() hm.Types { ts := hm.BorrowTypes(1); ts[0] = t.Of; return ts }
func (t TensorType) Normalize(k, v hm.TypeVarSet) (hm.Type, error) {
	var err error
	if t.Of, err = t.Of.Normalize(k, v); err != nil {
		return nil, err
	}

	return t, nil
}
func (t TensorType) Apply(sub hm.Subs) hm.Substitutable {
	t.Of = t.Of.Apply(sub).(hm.Type)
	return t
}

func (t TensorType) FreeTypeVar() hm.TypeVarSet {
	return t.Of.FreeTypeVar()
}
func (t TensorType) Eq(other hm.Type) bool {
	if ot, ok := other.(TensorType); ok {
		return ot.Of.Eq(t.Of) && ot.Dims == t.Dims
	}

	// if dt, ok := other.(Dtype); ok && t.d == 0 {
	// 	return t.of.Eq(dt)
	// }
	return false
}
