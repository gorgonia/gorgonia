package dual

/*
var dvpool = &sync.Pool{
	New: func() interface{} { return new(Dual) },
}

func borrowDV() *Dual { return dvpool.Get().(*Dual) }

// ReturnDV returns the values associated in a *Dual, then returns the *Dual to a pool
//
// USE WITH CAUTION.
func ReturnDV(dv *Dual) {
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
*/
