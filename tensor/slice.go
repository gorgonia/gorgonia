package tensor

// A Slice represents a slicing operation for a Tensor.
type Slice interface {
	Start() int
	End() int
	Step() int
}

type rs struct {
	start, end, step int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

// makeRS creates a ranged slice. It takes an optional step param.
func makeRS(start, end int, opts ...int) rs {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return rs{
		start: start,
		end:   end,
		step:  step,
	}
}

// ss is a single slice, representing this: [start:start+1:0]
type ss int

func (s ss) Start() int { return int(s) }
func (s ss) End() int   { return int(s) + 1 }
func (s ss) Step() int  { return 0 }
