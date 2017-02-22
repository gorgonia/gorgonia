package gorgonia

import "testing"

func TestStupid(t *testing.T) {
	g := NewGraph()
	n := newNode(WithType(Float64), In(g))
	op := newElemUnaryOp(negOpType, n)

	t.Logf("%v %d %s", op, op.unaryOpType(), op.ʘUnaryOperator)

	v := newF64(3.1415)
	rv, err := op.Do(v)
	t.Logf("%v, %v", rv, err)
}

func TestOpEquality(t *testing.T) {
	var op1, op2 Op
	g := NewGraph()
	a := NewScalar(g, Float64, WithValue(3.14))
	b := NewScalar(g, Float64, WithValue(6.28))
	op1 = newElemBinOp(addOpType, a, b)
	op2 = newElemBinOp(addOpType, a, b)

	if op1.Hashcode() != op2.Hashcode() {
		t.Error("oops")
	}

	op1 = maxOp{
		along: axes{0, 1},
		d:     2,
	}

	op2 = maxOp{
		along: axes{0, 1},
		d:     2,
	}

	if op1.Hashcode() != op2.Hashcode() {
		t.Error("oops")
	}

	op2 = sumOp{
		along: axes{0, 1},
		d:     2,
	}

	if op1.Hashcode() == op2.Hashcode() {
		t.Error("oops")
	}
}
