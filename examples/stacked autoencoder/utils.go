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

func hasOne(a []float64) bool {
	for _, v := range a {
		if v == 1.0 {
			return true
		}
	}
	return false
}

func avgF64s(a []float64) (retVal float64) {
	for _, v := range a {
		retVal += v
	}
	retVal /= float64(len(a))
	return
}
