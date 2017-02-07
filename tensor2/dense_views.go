package tensor

// a View is a *Tensor with customized strides. The reason for not splitting them up into different types is complicated
// this file contains all the methods that deals with Views

// Materialize takes a view, copies its data and puts it in a new *Tensor.
func (t *Dense) Materialize() Tensor {
	if !t.IsMaterializable() {
		return t
	}

	iter := NewFlatIterator(t.AP)

	newBack := makeArray(t.t, t.Shape().TotalSize())
	var newI int
	for i, err := iter.Next(); err == nil; i, err = iter.Next() {
		newBack.Set(newI, t.data.Get(i))
	}

	shape := t.Shape().Clone()
	retVal := recycledDense(t.t, shape, WithBacking(newBack))
	return retVal
}
