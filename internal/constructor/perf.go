package constructor

import "sync"

var tensorTypePool = &sync.Pool{
	New: func() interface{} { return new(TensorType) },
}

func borrowTensorType() *TensorType {
	return tensorTypePool.Get().(*TensorType)
}

// ReturnTensorType to the pool
func ReturnTensorType(t *TensorType) {
	switch t {
	case vecF64, vecF32:
		return
	case matF64, matF32:
		return
	case ten3F64, ten3F32:
		return
	}
	t.Of = nil
	t.Dims = 0
	tensorTypePool.Put(t)
}
