package tensori

import (
	"sync"

	"github.com/chewxy/gorgonia/tensor/types"
)

var habbo sync.Mutex
var useTensorPool bool = true

// tensorPool is a pool of *Tensor grouped by size. It's guarded by poolsClosed
var poolsClosed sync.RWMutex
var tensorPool map[int]*sync.Pool = make(map[int]*sync.Pool)

// UseTensorPool enables the use of a pool of *Tensors as provided in the package. This is the default option
func UseTensorPool() {
	habbo.Lock()
	defer habbo.Unlock()
	useTensorPool = true
}

// DontUseTensorPool makes sure the functions don't use the tensor pool provided.
// This is useful as certain applications don't lend themselves well to use of the pool.
// Examples of such applications would be one where many tensors of wildly different sizes are created all the time.
func DontUseTensorPool() {
	habbo.Lock()
	defer habbo.Unlock()
	useTensorPool = false
}

func newSyncPool(size int) *sync.Pool {
	pool := new(sync.Pool)
	l := size
	pool.New = func() interface{} {
		return newTensor(l)
	}

	poolsClosed.Lock()
	// check once more that before the lock was acquired, that nothing else had written to that key
	if p, ok := tensorPool[size]; !ok {
		tensorPool[size] = pool
	} else {
		pool = p
	}
	poolsClosed.Unlock()
	return pool
}

func BorrowTensor(size int) *Tensor {
	if !useTensorPool {
		return NewTensor(WithShape(size, 1))
	}

	poolsClosed.RLock()
	pool, ok := tensorPool[size]
	poolsClosed.RUnlock()

	if !ok {
		pool = newSyncPool(size)
	}

	return pool.Get().(*Tensor)
}

func ReturnTensor(t *Tensor) {
	if !useTensorPool {
		return
	}

	// important: don't use .Size() because it may not be accurate (views and such)
	size := len(t.data)

	poolsClosed.RLock()
	pool, ok := tensorPool[size]
	poolsClosed.RUnlock()

	if !ok {
		pool = newSyncPool(size)
	}

	if t.old != nil {
		types.ReturnAP(t.old)
		t.old = nil
	}

	if t.transposeWith != nil {
		types.ReturnInts(t.transposeWith)
		t.transposeWith = nil
	}

	pool.Put(t)
}
