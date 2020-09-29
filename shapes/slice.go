package shapes

import "fmt"

// Slice represents a slicing range.
type Slice interface {
	Start() int
	End() int
	Step() int
}

// Sli is a shape expression representing a slicing range. Coincidentally, Sli also implements Slice.
//
// A Sli is a shape expression but it doesn't stand alone - resolving it will yield an error.
type Sli struct {
	start, end, step int
}

func (s Sli) isExpr() {}

func (s Sli) apply(ss substitutions) substitutable { return s }
func (s Sli) freevars() varset                     { return nil }

// Exprs returns nil because we want a Sli to be treated as a monolithic expression term with nothing to unify on the inside.
func (s Sli) subExprs() []substitutableExpr { return nil }

// Format allows Sli to implement fmt.Formmatter
func (s Sli) Format(st fmt.State, r rune) {
	fmt.Fprintf(st, "[%d", s.start)
	if s.end-s.start > 1 {
		fmt.Fprintf(st, ":%d", s.end)
	}
	if s.step > 1 {
		fmt.Fprintf(st, "~:%d", s.step)
	}
	st.Write([]byte("]"))
}

/* Sli implements Slice */

// Start returns the start of the slicing range
func (s Sli) Start() int { return s.start }

// End returns the end of the slicing range
func (s Sli) End() int { return s.end }

// Step returns the steps/jumps to make in the slicing range.
func (s Sli) Step() int { return s.step }

// S creates a Slice
func S(start int, opt ...int) Slice {
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
	return &Sli{
		start: start,
		end:   end,
		step:  step,
	}
}

// toSli creates a Sli from a Slice.
func toSli(s Slice) Sli {
	if ss, ok := s.(Sli); ok {
		return ss
	}
	if ss, ok := s.(*Sli); ok {
		return *ss
	}
	return Sli{s.Start(), s.End(), s.Step()}
}
