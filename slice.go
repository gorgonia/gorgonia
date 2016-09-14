package gorgonia

// sli is slice. It's named sli to prevent confusion over naming
type sli struct {
	start, end int
}

// Slice. End is optional. If end isn't provided, end = start+1
func S(start int, opt ...int) sli {
	var end int
	if len(opt) > 0 {
		end = opt[0]
	} else {
		end = start + 1
	}

	return sli{
		start: start,
		end:   end,
	}
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
