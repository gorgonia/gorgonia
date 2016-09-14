package gorgonia

import "testing"

func TestStupid(t *testing.T) {
	g := NewGraph()
	n := newNodeFromPool(withType(Float64), withGraph(g))
	op := newElemUnaryOp(negOpType, n)

	t.Logf("%p %d %s", op, op.unaryOpType(), op.ʘUnaryOperator)

	v := NewScalarValue(3.1415)
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
