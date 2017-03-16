package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
)

var (
	// Represents the types that Nodes can take in Gorgonia

	Float64 = tensor.Float64
	Float32 = tensor.Float32
	Int     = tensor.Int
	Int64   = tensor.Int64
	Int32   = tensor.Int32
	Byte    = tensor.Uint8
	Bool    = tensor.Bool

	Ptr = tensor.UnsafePointer // equivalent to interface{}. Ugh Ugh Ugh
)

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
	if ot, ok := other.(TensorType); ok {
		return ot.Of.Eq(t.Of) && ot.Dims == t.Dims
	}
	return false
}
