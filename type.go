package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor/types"
)

// functionType is a type constructor that builds function types
type functionType struct {
	n  *typeVariable
	ts [2]Type // from → to
}

func newFunctionType(params ...Type) *functionType {
	if len(params) < 2 {
		panic(fmt.Sprintf("Needs more than 2 params to make a function. Got %v", params))
	}

	t := borrowFnType()
	t.ts[0] = params[0]
	if len(params) == 2 {
		t.ts[1] = params[1]
		return t
	}

	t.ts[1] = newFunctionType(params[1:]...)
	return t
}

func (mt *functionType) types() Types            { return Types(mt.ts[:]) }
func (mt *functionType) name() *typeVariable     { return mt.n }
func (mt *functionType) isScalar() bool          { return false }
func (mt *functionType) dims() int               { return -1 }
func (mt *functionType) String() string          { return fmt.Sprintf("%s → %s", mt.ts[0], mt.ts[1]) }
func (mt *functionType) setName(n *typeVariable) { mt.n = n }
func (mt *functionType) setTypes(ts ...Type) {
	if len(ts) != 2 {
		panic("Impossible type")
	}

	mt.ts[0] = ts[0]
	mt.ts[1] = ts[1]
}

func (t *functionType) retType() Type {
	if r, ok := prune(t.ts[1]).(*functionType); ok {
		return r.retType()
	}
	return t.ts[1]
}

/*Tensor Type*/

// a TensorType is a type constructor for tensors.
//
// Think of it as  something like this:
//		data Tensor a = Tensor d a
//
// The shape of the Tensor is not part of TensorType.
// Shape checking is relegated to the dynamic part of the program run
type TensorType struct {
	d     int // dims
	shape types.Shape

	of Type
	n  *typeVariable
}

func fromTensorType(t *TensorType, tv *typeVariable) *TensorType {
	retVal := newTensorType(t.d, tv)
	retVal.shape = t.shape.Clone()
	retVal.n = t.n
	return retVal
}

func newTensorType(dims int, typ Type) *TensorType {
	t := new(TensorType)
	t.d = dims
	t.of = typ

	if _, ok := t.of.(Dtype); ok {
		scalarOrTensor.addInstance(t)
		arithable.addInstance(t)
		summable.addInstance(t)
	}
	return t
}

func (t *TensorType) types() Types {
	ts := borrowTypes1()
	ts[0] = t.of
	return ts
}

func (t *TensorType) name() *typeVariable { return t.n }
func (t *TensorType) isScalar() bool      { return false }
func (t *TensorType) dims() int           { return t.d }

func (t *TensorType) String() string {
	switch t.d {
	case 1:
		return fmt.Sprintf("Vector %s", t.of)
	case 2:
		return fmt.Sprintf("Matrix %s", t.of)
	}
	return fmt.Sprintf("Tensor-%d %s", t.d, t.of)
}

func (t *TensorType) setName(n *typeVariable) { t.of = n }

func (t *TensorType) setTypes(ts ...Type) {
	if len(ts) != 1 {
		panic("Impossible type")
	}

	t.of = ts[0]
}
