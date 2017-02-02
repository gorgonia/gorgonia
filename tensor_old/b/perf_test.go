package tensorb

import (
	"testing"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestTensorPool(t *testing.T) {
	assert := assert.New(t)
	T := BorrowTensor(4)

	assert.Equal(types.Shape{4}, T.Shape())
	assert.Nil(T.viewOf)
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)

	// modify the tensor
	T.transposeWith = []int{1, 2, 3, 4}
	T.Reshape(2, 2)
	T.old = new(types.AP)

	// return it to the pool, but because variable T is still defined in this func
	// and this func hasn't exited, the reference to the *Tensor is still held
	// therefore won't be GC'd away.
	ReturnTensor(T)
	T = BorrowTensor(4)

	assert.Equal(types.Shape{2, 2}, T.Shape())
	assert.Nil(T.viewOf)
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)

	// test returning of tensor that doesn't yet exist in the pool
	delete(tensorPool, 6) // makes sure that this is empty
	T = NewTensor(WithShape(2, 3))

	// now we turn off the pool
	DontUseTensorPool()

	ptr := uintptr(unsafe.Pointer(T))
	ReturnTensor(T) // does nothing. T will be GC'd away
	T = nil         // go gc go!

	T2 := BorrowTensor(4)
	ptr2 := uintptr(unsafe.Pointer(T2))

	assert.NotEqual(ptr, ptr2)

}
