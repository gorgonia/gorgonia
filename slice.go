package gorgonia

// type SliceValue interface {
// 	types.Slice
// 	Value
// }

// sli is slice. It's named sli to prevent confusion over naming
type sli struct {
	start, end, step int
}

// S creates a types.Slice.
// end is optional. It should be passed in as the first param of the optionals.
// step is optional. It should be passed in as the second param of the optionals.
//
// Default end is start+1. Default step is 1, unless end == step+1, then it defaults to 0
func S(start int, opt ...int) sli {
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

	return sli{
		start: start,
		end:   end,
		step:  step,
	}
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return s.step }

// sli implements SliceValue such that a slice can be passed into Let
// func (s sli) Shape() types.Shape             { return nil }
// func (s sli) Size() int                      { return -1 }
// func (s sli) Data() interface{}              { return s }
// func (s sli) Format(state fmt.State, c rune) { fmt.Fprintf(state, "%d:%d:%d", s.start, s.end, s.step) }
