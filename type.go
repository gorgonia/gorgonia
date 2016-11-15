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

func (t Dtype) Name() string                  { return t.String() }
func (t Dtype) Contains(hm.TypeVariable) bool { return false }
func (t Dtype) Eq(other hm.Type) bool {
	if odt, ok := other.(Dtype); ok {
		return odt == t
	}
	return false
}

func (t Dtype) Format(state fmt.State, c rune)     { state.Write([]byte(t.String())) }
func (t Dtype) Types() hm.Types                    { return nil }
func (t Dtype) Clone() hm.TypeOp                   { return t }
func (t Dtype) Replace(hm.Type, hm.Type) hm.TypeOp { return t }
func (t Dtype) IsConstant() bool                   { return true }

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
	t := TensorType{
		d:  dims,
		of: typ,
	}

	if _, ok := t.of.(Dtype); ok {
		scalarOrTensor.AddInstance(t)
		arithable.AddInstance(t)
		summable.AddInstance(t)
	}
	return t
}

func (t TensorType) Name() string                     { return "Tensor" }
func (t TensorType) Contains(tv hm.TypeVariable) bool { return t.of.Eq(tv) }
func (t TensorType) Eq(other hm.Type) bool {
	if ott, ok := other.(TensorType); ok {
		if t.of.Eq(ott.of) && t.d == ott.d {
			return true
		}
	}
	return false
}

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
func (t TensorType) String() string { return fmt.Sprintf("%v", t) }

func (t TensorType) Types() hm.Types { return hm.Types{t.of} }

func (t TensorType) Clone() hm.TypeOp {
	var of hm.Type
	switch tt := t.of.(type) {
	case hm.TypeVariable:
		of = hm.NewTypeVar(tt.Name())
	case hm.TypeOp:
		of = tt.Clone()
	default:
		panic("WTF?")
	}

	return TensorType{
		d:  t.d,
		of: of,
	}
}

func (t TensorType) Replace(what, with hm.Type) hm.TypeOp {
	switch tt := t.of.(type) {
	case hm.TypeVariable:
		if tt.Eq(what) {
			t.of = with
		}
	case hm.TypeConst:
		// do nothing
	case hm.TypeOp:
		if tt.Eq(what) {
			t.of = with
		} else {
			t.of = tt.Replace(what, with)
		}
	default:
		panic("WTF")
	}
	return t
}
