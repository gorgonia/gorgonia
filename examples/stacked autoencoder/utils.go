package main

type sli struct {
	start, end, step int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return s.step }

func s(start int) sli {
	return sli{
		start: start,
		end:   start + 1,
		step:  0,
	}
}
