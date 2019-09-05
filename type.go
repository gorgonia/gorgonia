package gorgonia

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

var (
	// Represents the types that Nodes can take in Gorgonia

	// Float64 ...
	Float64 = tensor.Float64
	// Float32 ...
	Float32 = tensor.Float32
	// Int ...
	Int = tensor.Int
	// Int64 ...
	Int64 = tensor.Int64
	// Int32 ...
	Int32 = tensor.Int32
	// Byte ...
	Byte = tensor.Uint8
	// Bool ...
	Bool = tensor.Bool

	// Ptr is equivalent to interface{}. Ugh Ugh Ugh
	Ptr = tensor.UnsafePointer

	vecF64  = &TensorType{Dims: 1, Of: tensor.Float64}
	vecF32  = &TensorType{Dims: 1, Of: tensor.Float32}
	matF64  = &TensorType{Dims: 2, Of: tensor.Float64}
	matF32  = &TensorType{Dims: 2, Of: tensor.Float32}
	ten3F64 = &TensorType{Dims: 3, Of: tensor.Float64}
	ten3F32 = &TensorType{Dims: 3, Of: tensor.Float32}

	// removes the need for type checking
	f64T = tensor.Float64 // hm.Type
	f32T = tensor.Float32 // hm.Type
)

var acceptableDtypes = [...]tensor.Dtype{tensor.Float64, tensor.Float32, tensor.Int, tensor.Int64, tensor.Int32, tensor.Byte, tensor.Bool}

/*Tensor Type*/

// TensorType is a type constructor for tensors.
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

func makeFromTensorType(t TensorType, tv hm.TypeVariable) TensorType {
	return makeTensorType(t.Dims, tv)
}

func makeTensorType(dims int, typ hm.Type) TensorType {
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

// String implements fmt.Stringer and runtime.Stringer. Satisfies the hm.Type interface.
func (t TensorType) String() string { return fmt.Sprintf("%v", t) }

// Types returns a list of types that TensorType contains - in this case, the type of Tensor (float64, float32, etc). Satisfies the hm.Type interface.
func (t TensorType) Types() hm.Types { ts := hm.BorrowTypes(1); ts[0] = t.Of; return ts }

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

// Eq is the equality function of this type. The type of Tensor has to be the same, and for now, only the dimensions are compared.
// Shape may be compared in the future for tighter type inference. Satisfies the hm.Type interface.
func (t TensorType) Eq(other hm.Type) bool {
	switch ot := other.(type) {
	case TensorType:
		return t.Of.Eq(ot.Of) && t.Dims == ot.Dims
	case *TensorType:
		return t.Of.Eq(ot.Of) && t.Dims == ot.Dims
	}
	return false
}
