package tensor

type funcOpt struct {
	reuse  Tensor
	incr   Tensor
	unsafe bool
	same   bool
}

func parseFuncOpts(opts ...FuncOpt) *funcOpt {
	retVal := new(funcOpt)
	for _, opt := range opts {
		opt(retVal)
	}
	return retVal
}

// incReuse returns whether a reuse tensor is to be used as the incr Tensor
func (fo *funcOpt) incrReuse() (Tensor, bool) {
	if fo.incr != nil {
		return fo.incr, true
	}
	return fo.reuse, false
}

func (fo *funcOpt) safe() bool { return !fo.unsafe }
