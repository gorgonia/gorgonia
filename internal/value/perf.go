package value

import (
	"sync"

	"gorgonia.org/tensor"
)

// handles Returning of value.Values

var dvpool = &sync.Pool{
	New: func() interface{} { return new(DualValue) },
}

// BorrowDV get a DualValue from the pool
func BorrowDV() *DualValue { return dvpool.Get().(*DualValue) }

// ReturnDV returns the DualValue to the pool
func ReturnDV(dv *DualValue) {
	returnValue(dv.D)
	returnValue(dv.Value)
	// if dvdT, ok := dv.d.(tensor.Tensor); ok {
	// 	returnTensor(dvdT)
	// }
	// if dvvT, ok := dv.Value.(tensor.Tensor); ok {
	// 	returnTensor(dvvT)
	// }

	dv.D = nil
	dv.Value = nil
	dvpool.Put(dv)
}
func returnValue(v Value) {
	if t, ok := v.(tensor.Tensor); ok {
		tensor.ReturnTensor(t)
		//	returnTensor(t)
	}
}
