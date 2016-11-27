package gorgonia

import (
	"fmt"

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

/*Tensor Type*/

// a TensorType is a type constructor for tensors.
//
// Think of it as  something like this:
//		data Tensor a = Tensor d a
//
// The shape of the Tensor is not part of TensorType.
// Shape checking is relegated to the dynamic part of the program run
type TensorType struct {
	d int // dims

	of hm.Type
}

func fromTensorType(t TensorType, tv hm.TypeVariable) TensorType {
	retVal := newTensorType(t.d, tv)
	return retVal
}

func newTensorType(dims int, typ hm.Type) TensorType {
	return TensorType{
		d:  dims,
		of: typ,
	}
}

func (t TensorType) Name() string { return "Tensor" }

func (t TensorType) Format(state fmt.State, c rune) {
	if state.Flag('#') {
		fmt.Fprintf(state, "Tensor-%d %#v", t.d, t.of)
	} else {
		switch t.d {
		case 1:
			fmt.Fprintf(state, "Vector %v", t.of)
		case 2:
			fmt.Fprintf(state, "Matrix %v", t.of)
		default:
			fmt.Fprintf(state, "Tensor-%d %v", t.d, t.of)
		}
	}
}
func (t TensorType) String() string  { return fmt.Sprintf("%v", t) }
func (t TensorType) Types() hm.Types { ts := hm.BorrowTypes(1); ts[0] = t.of; return ts }
func (t TensorType) Normalize(k, v hm.TypeVarSet) (hm.Type, error) {
	var err error
	if t.of, err = t.of.Normalize(k, v); err != nil {
		return nil, err
	}

	return t, nil
}
func (t TensorType) Apply(sub hm.Subs) hm.Substitutable {
	t.of = t.of.Apply(sub).(hm.Type)
	return t
}

func (t TensorType) FreeTypeVar() hm.TypeVarSet {
	return t.of.FreeTypeVar()
}
func (t TensorType) Eq(other hm.Type) bool {
	if ot, ok := other.(TensorType); ok {
		return ot.of.Eq(t.of) && ot.d == t.d
	}

	// if dt, ok := other.(Dtype); ok && t.d == 0 {
	// 	return t.of.Eq(dt)
	// }
	return false
}
