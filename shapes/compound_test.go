package shapes

import "testing"

func TestSubjectTo(t *testing.T) {
	st := SubjectTo{
		OpType: Eq,
		A:      UnaryOp{Dims, Shape{2, 3, 4}},
		B:      UnaryOp{Dims, Shape{3, 4, 5}},
	}
	t.Logf("%v", st)
}
