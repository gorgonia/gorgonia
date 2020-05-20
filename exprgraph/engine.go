package exprgraph

import "gorgonia.org/tensor"

func (e *Graph) MatMul(a, b, preallocated tensor.Tensor) error {
	aEng := a.Engine().(*Graph) // TODO ERR
	bEng := b.Engine().(*Graph) // TODO ERR
	cEng := preallocated.Engine().(*Graph)

	aName := aEng.nameOf(a)
	bName := bEng.nameOf(b)

	cEng.idOrInsert(preallocated) // TODO what if preallocated already exists as a node? (i.e WithReuse was called)
	cName := aName + "×" + bName
	cEng.name(preallocated, cName)

	// TODO: op

	return e.StdEng.MatMul(a, b, preallocated)
}

// Add performs a + b
func (e *Graph) Add(a tensor.Tensor, b tensor.Tensor, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	aEng := a.Engine().(*Graph) // TODO ERR
	bEng := b.Engine().(*Graph) // TODO ERR

	aName := aEng.nameOf(a)
	bName := bEng.nameOf(b)
	cName := aName + "+" + bName
	retVal, err := e.StdEng.Add(a, b, opts...)
	if err != nil {
		return nil, err
	}
	cEng := retVal.Engine().(*Graph)
	cEng.name(retVal, cName)
	return retVal, err
}

// AddScalar adds a scalar to the tensor. leftTensor indicates if the tensor is the left operand.
// Whether or not the input tensor is clobbered is left to the implementation
func (e *Graph) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	panic("not implemented") // TODO: Implement
}
