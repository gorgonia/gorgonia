package dual

import (
	"sync"

	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

var dvpool = &sync.Pool{
	New: func() interface{} { return new(Dual) },
}

func borrowDV() *Dual { return dvpool.Get().(*Dual) }

func returnDV(dv *Dual) {
	returnValue(dv.d)
	returnValue(dv.Value)
	dv.d = nil
	dv.Value = nil
	dvpool.Put(dv)
}

func returnValue(v values.Value) {
	if t, ok := v.(tensor.Tensor); ok {
		tensor.ReturnTensor(t)
	}
}
