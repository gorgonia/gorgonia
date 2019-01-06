package perf

import (
	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/internal/constructor"
	"gorgonia.org/tensor"
)

// ReturnType ...
func ReturnType(t hm.Type) {
	switch tt := t.(type) {
	case *constructor.TensorType:
		constructor.ReturnTensorType(tt)
	case constructor.TensorType:
		// do nothing
	case tensor.Dtype:
		// do nothing
	case *hm.FunctionType:
		hm.ReturnFnType(tt)
	}
}
