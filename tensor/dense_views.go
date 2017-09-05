package tensor

// a View is a *Tensor with customized strides. The reason for not splitting them up into different types is complicated
// this file contains all the methods that deals with Views

// Materialize takes a view, copies its data and puts it in a new *Tensor.
func (t *Dense) Materialize() Tensor {
	if !t.IsMaterializable() {
		return t
	}

	retVal := recycledDense(t.t, t.shape.Clone())
	copyDenseIter(retVal, t, nil, nil)
	retVal.e = t.e
	retVal.oe = t.oe
	return retVal
}
