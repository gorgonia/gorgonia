package types

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/dtype"
)

var (
	// Represents the types that Nodes can take in Gorgonia

	Float64 = dtype.Float64
	Float32 = dtype.Float32
	Int     = dtype.Int
	Int64   = dtype.Int64
	Int32   = dtype.Int32
	Byte    = dtype.Uint8
	Bool    = dtype.Bool

	Ptr = dtype.UnsafePointer // equivalent to interface{}. Ugh Ugh Ugh

	vecF64  = &TensorType{Dims: 1, Of: dtype.Float64}
	vecF32  = &TensorType{Dims: 1, Of: dtype.Float32}
	matF64  = &TensorType{Dims: 2, Of: dtype.Float64}
	matF32  = &TensorType{Dims: 2, Of: dtype.Float32}
	ten3F64 = &TensorType{Dims: 3, Of: dtype.Float64}
	ten3F32 = &TensorType{Dims: 3, Of: dtype.Float32}

	// removes the need for type checking
	f64T hm.Type = dtype.Float64
	f32T hm.Type = dtype.Float32
)

/*Tensor Type*/

// TensorType is a type constructor for tensors.
//
// Think of it as  something like this:
//
//	data Tensor a = Tensor d a
//
// The shape of the Tensor is not part of TensorType.
// Shape checking is relegated to the dynamic part of the program run
type TensorType struct {
	Dims int // dims

	Of hm.Type
}

// MakeTensorTypeLike creates a new TensorType like  the provided TensorType
func MakeTensorTypeLike(t TensorType, tv hm.TypeVariable) TensorType {
	return MakeTensorType(t.Dims, tv)
}

// MakeTensorType makes a TensorType
func MakeTensorType(dims int, typ hm.Type) TensorType {
	return TensorType{
		Dims: dims,
		Of:   typ,
	}
}

func newTensorType(dims int, typ hm.Type) *TensorType {
	switch {
	case dims == 1 && typ == f64T:
		return vecF64
	case dims == 1 && typ == f32T:
		return vecF32
	case dims == 2 && typ == f64T:
		return matF64
	case dims == 2 && typ == f32T:
		return matF32
	case dims == 3 && typ == f64T:
		return ten3F64
	case dims == 3 && typ == f32T:
		return ten3F32
	}
	t := borrowTensorType()
	t.Dims = dims
	t.Of = typ
	return t
}

// Name returns the name of the type, which will always be "Tensor". Satisfies the hm.Type interface.
func (t TensorType) Name() string { return "Tensor" }

// Format implements fmt.Formatter. It is also required for the satisfication the hm.Type interface.
func (t TensorType) Format(state fmt.State, c rune) {
	if state.Flag('#') {
		fmt.Fprintf(state, "Tensor-%d %#v", t.Dims, t.Of)
	} else {
		switch {
		case t.Dims < 0:
			fmt.Fprintf(state, "Tensor-n(#%d) %v", -t.Dims, t.Of) // negative numbers are "variables" for dims. THIS IS A HACK.
		case t.Dims == 0:
			fmt.Fprintf(state, "%v", t.Of)
		case t.Dims == 1:
			fmt.Fprintf(state, "Vector %v", t.Of)
		case t.Dims == 2:
			fmt.Fprintf(state, "Matrix %v", t.Of)
		default:
			fmt.Fprintf(state, "Tensor-%d %v", t.Dims, t.Of)
		}
	}
}

// String implements fmt.Stringer and runtime.Stringer. Satisfies the hm.Type interface.
func (t TensorType) String() string { return fmt.Sprintf("%v", t) }

// Types returns a list of types that TensorType contains - in this case, the type of Tensor (float64, float32, etc). Satisfies the hm.Type interface.
func (t TensorType) Types() hm.Types {
	ts := hm.BorrowTypes(1)
	ts[0] = t.Of
	return ts
}

// Normalize normalizes the type variable names (if any) in the TensorType. Satisfies the hm.Type interface.
func (t TensorType) Normalize(k, v hm.TypeVarSet) (hm.Type, error) {
	var err error
	if t.Of, err = t.Of.Normalize(k, v); err != nil {
		return nil, err
	}

	return t, nil
}

// Apply applies the substitutions on the types. Satisfies the hm.Type interface.
func (t TensorType) Apply(sub hm.Subs) hm.Substitutable {
	t.Of = t.Of.Apply(sub).(hm.Type)
	return t
}

// FreeTypeVar returns any free (unbound) type variables in this type. Satisfies the hm.Type interface.
func (t TensorType) FreeTypeVar() hm.TypeVarSet {
	return t.Of.FreeTypeVar()
}

// Eq is the equality function of this type.
// Eq allows TensorType to satisfy the hm.Type interface.
func (t TensorType) Eq(other hm.Type) bool {
	switch ot := other.(type) {
	case TensorType:
		if t.Dims < 0 || ot.Dims < 0 {
			return t.Of.Eq(ot.Of)
		}
		return t.Of.Eq(ot.Of) && t.Dims == ot.Dims
	case *TensorType:
		if t.Dims < 0 || ot.Dims < 0 {
			return t.Of.Eq(ot.Of)
		}
		return t.Of.Eq(ot.Of) && t.Dims == ot.Dims
	}
	if t.Dims == 0 {
		return t.Of.Eq(other)
	}

	return false
}

// Canonical returns the canonical type. This is because TensorType can represent a scalar type as well.
func (t TensorType) Canonical() hm.Type {
	if t.Dims == 0 {
		return t.Of
	}
	return t
}
