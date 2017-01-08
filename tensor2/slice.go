package tensor

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
