package tensorb

import "github.com/chewxy/gorgonia/tensor/types"

// a View is a *Tensor with customized strides. The reason for not splitting them up into different types is complicated
// this file contains all the methods that deals with Views

// Materialize takes a view, copies its data and puts it in a new *Tensor.
// The reason why it returns a types.Tensor is to fulfil the types.Tensor interface. Not ideal, I know, but for now it works
func (t *Tensor) Materialize() (retVal types.Tensor) {
	if !t.IsMaterializable() {
		return t
	}

	iter := types.NewFlatIterator(t.AP)

	newBack := make([]bool, t.Shape().TotalSize())
	newBack = newBack[:0]
	for i, err := iter.Next(); err == nil; i, err = iter.Next() {
		newBack = append(newBack, t.data[i])
	}

	retVal = NewTensor(WithShape(t.Shape()...), WithBacking(newBack))

	return
}
