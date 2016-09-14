package tensorb

import "github.com/chewxy/gorgonia/tensor/types"

// a View is a *Tensor with customized strides. The reason for not splitting them up into different types is complicated
// this file contains all the methods that deals with Views

// a forward-only iterator
type iterator struct {
	*Tensor

	// state
	lastIndex int
	track     types.Shape
	done      bool
}

func newIterator(t *Tensor) *iterator {
	return &iterator{
		Tensor: t,

		lastIndex: -1,
		track:     make(types.Shape, len(t.oshape())),
	}
}

func (it *iterator) next() (int, error) {
	if it.viewOf == nil {
		it.lastIndex++
		if it.lastIndex >= len(it.data) {
			it.done = true
			return -1, noopError{}
		}
		return it.lastIndex, nil
	}

	if it.done {
		return -1, noopError{}
	}

	defer func() {
		if it.IsScalar() {
			it.done = true
			return
		}

		for d := len(it.oshape()) - 1; d >= 0; d-- {
			if d == 0 && it.track[0]+1 >= it.oshape()[0] {
				it.done = true
				break
			}

			if it.track[d] < it.oshape()[d]-1 {
				it.track[d]++
				break
			}
			// overflow
			it.track[d] = 0
		}
	}()

	retVal, err := it.at(it.track...)
	it.lastIndex = retVal
	return retVal, err
}

// Materialize takes a view, copies its data and puts it in a new *Tensor.
// The reason why it returns a types.Tensor is to fulfil the types.Tensor interface. Not ideal, I know, but for now it works
func (t *Tensor) Materialize() (retVal types.Tensor) {
	if t.viewOf == nil {
		return t
	}

	iter := newIterator(t)

	var newBack []bool
	for i, err := iter.next(); err == nil; i, err = iter.next() {
		newBack = append(newBack, t.data[i])
	}

	retVal = NewTensor(WithShape(t.Shape()...), WithBacking(newBack))
	return
}
