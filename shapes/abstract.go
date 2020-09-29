package shapes

import "fmt"

// Abstract is an abstract shape
type Abstract []Sizelike

func (a Abstract) Cons(other Conser) (retVal Conser) {
	switch ot := other.(type) {
	case Shape:
		l := len(a)
		r := append(a, make(Abstract, len(ot))...)
		r = r[:l]
		for _, v := range ot {
			r = append(r, Size(v))
		}
		retVal = r
	case Abstract:
		retVal = append(a, ot...)
	}
	return retVal
}

func (a Abstract) isConser() {}

func (a Abstract) ToShape() (s Shape, ok bool) {
	s = make(Shape, len(a)) // TODO: perf - borrow
	for i := range a {
		sz, ok := a[i].(Size)
		if !ok {
			return nil, ok
		}
		s[i] = int(sz)
	}
	return s, true
}

func (a Abstract) isExpr() {}

func ToShape(a Abstract) Shape { panic("NYI") }

func (a Abstract) Dims() int {
	panic("not implemented") // TODO: Implement
}

func (a Abstract) TotalSize() int {
	panic("not implemented") // TODO: Implement
}

func (a Abstract) DimSize(dim int) (Sizelike, error) {
	panic("not implemented") // TODO: Implement
}

func (a Abstract) T(axes ...Axis) (newShape Shapelike, err error) {
	panic("not implemented") // TODO: Implement
}

func (a Abstract) S(slices ...Slice) (newShape Shapelike, err error) {
	panic("not implemented") // TODO: Implement
}

func (a Abstract) Repeat(axis Axis, repeats ...int) (newShape Shapelike, finalRepeats []int, size int, err error) {
	panic("not implemented") // TODO: Implement
}

func (a Abstract) Concat(axis Axis, others ...Shapelike) (newShape Shapelike, err error) {
	panic("not implemented") // TODO: Implement
}

// Format implements fmt.Formatter, and formats a shape nicely
func (s Abstract) Format(st fmt.State, r rune) {
	switch r {
	case 'v', 's':
		st.Write([]byte("("))
		for i, v := range s {
			switch vt := v.(type) {
			case Size:
				fmt.Fprintf(st, "%d", int(vt))
			case Var:
				fmt.Fprintf(st, "%c", rune(vt))
			case BinOp:
				fmt.Fprintf(st, "%v", vt)
			}
			if i < len(s)-1 {
				st.Write([]byte(", "))
			}
		}
		st.Write([]byte(")"))
	default:
		fmt.Fprintf(st, "%v", []Sizelike(s))
	}
}

func (s Abstract) apply(ss substitutions) substitutable {
	retVal := make(Abstract, len(s))
	copy(retVal, s)
	for i, a := range s {
		if v, ok := a.(Var); ok {
			for _, sub := range ss {
				if v == sub.For {
					retVal[i] = sub.Sub.(Sizelike)
					break
				}
			}
		}
	}
	return retVal
}

func (s Abstract) freevars() (retVal varset) {
	for _, a := range s {
		if v, ok := a.(Var); ok {
			retVal = append(retVal, v)
		}
	}
	return unique(retVal)
}

func (s Abstract) subExprs() (retVal []substitutableExpr) {
	for i := range s {
		retVal = append(retVal, s[i].(substitutableExpr))
	}
	return retVal
}
