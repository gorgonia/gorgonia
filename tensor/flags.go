package tensor

// OpOpt are the options used to call ops
type OpOpt struct {
	reuse  Tensor
	incr   Tensor
	unsafe bool
	same   bool
	t      Dtype
}

// ParseFuncOpts parses a list of FuncOpt into a single unified method call structure.
func ParseFuncOpts(opts ...FuncOpt) *OpOpt {
	retVal := new(OpOpt)
	for _, opt := range opts {
		opt(retVal)
	}
	return retVal
}

// Incr returns the tensor to be incremented in the call. Can be nil.
func (fo *OpOpt) Incr() (Tensor) {return fo.incr}

// Reuse returns the tensor to be reused in the call. Can be nil.
func (fo *OpOpt) Reuse() Tensor {return fo.reuse}

// IncReuse returns whether a reuse tensor is to be used as the incr Tensor
func (fo *OpOpt) IncrReuse() (Tensor, bool) {
	if fo.incr != nil {
		return fo.incr, true
	}
	return fo.reuse, false
}

// Safe signals if the op is to be done safely
func (fo *OpOpt) Safe() bool { return !fo.unsafe }

// Same signals if the op is to return the same type as its inputs
func (fo *OpOpt) Same() bool {return fo.same}


// As returns the dtype of the return value of the method call. 
// For example:
//		a.Lt(b, As(Bool))
// indicates that the result of the `Lt()` should be a Tensor of Bool.
//
// Another example:
//		a.Add(b, As(Int))
// indicates that the result of `Add()` should be converted to a Tensor of Int.
// Note that this function is not yet supported in most operations.
func (fo *OpOpt) As() Dtype {return fo.t}