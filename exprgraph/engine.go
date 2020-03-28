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
