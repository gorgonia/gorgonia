package types

import (
	"sync"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

var tensorTypePool = &sync.Pool{
	New: func() interface{} { return new(TensorType) },
}

func borrowTensorType() *TensorType {
	return tensorTypePool.Get().(*TensorType)
}

func returnTensorType(t *TensorType) {
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

// ReturnType returns a Type to an internal pool.
// USE WITH CAUTION.
func ReturnType(t hm.Type) {
	switch tt := t.(type) {
	case *TensorType:
		returnTensorType(tt)
	case TensorType:
		// do nothing
	case tensor.Dtype:
		// do nothing
	case *hm.FunctionType:
		hm.ReturnFnType(tt)
	}
}
