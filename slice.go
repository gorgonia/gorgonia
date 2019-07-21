package gorgonia

import "gorgonia.org/tensor"

// sli is slice. It's named sli to prevent confusion over naming
type sli struct {
	start, end, step int
}

// S creates a tensor.Slice.
// end is optional. It should be passed in as the first param of the optionals.
// step is optional. It should be passed in as the second param of the optionals.
//
// Default end is start+1. Default step is 1, unless end == step+1, then it defaults to 0
func S(start int, opt ...int) tensor.Slice {
	var end, step int
	if len(opt) > 0 {
		end = opt[0]
	} else {
		end = start + 1
	}

	step = 1
	if len(opt) > 1 {
		step = opt[1]
	} else if end == start+1 {
		step = 0
	}

	return &sli{
		start: start,
		end:   end,
		step:  step,
	}
}

func (s *sli) Start() int { return s.start }
func (s *sli) End() int   { return s.end }
func (s *sli) Step() int  { return s.step }
