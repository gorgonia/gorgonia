package gorgonia

import (
	"io/ioutil"
	"testing"
)

func TestLogStabilization(t *testing.T) {
	g := NewGraph()

	// log(a+1)
	x := NewVector(g, Float64, WithName("x"), WithShape(2))
	p := Must(Add(x, onef64))
	lp := Must(Log(p))
	if lp.children[0] != x {
		t.Error("Oops.")
		ioutil.WriteFile("log(a+1).dot", []byte(lp.ToDot()), 0644)
	}

	// log(1+a)
	p = Must(Add(onef64, x))
	lp = Must(Log(p))
	if lp.children[0] != x {
		t.Error("Oops.")
		ioutil.WriteFile("log(1+a).dot", []byte(lp.ToDot()), 0644)
	}

	//log(1-a)
	m := Must(Sub(onef64, x))
	lp = Must(Log(m))
	if euo, ok := lp.children[0].op.(elemUnaryOp); !ok {
		t.Error("Oops.")
	} else {
		if euo.unaryOpType() != negOpType {
			t.Error("Expected Neg Op")
		}

		if lp.children[0].children[0] != x {
			t.Error("Oops.")
		}
	}

	if t.Failed() {
		ioutil.WriteFile("log(1-a).dot", []byte(lp.ToDot()), 0644)
	}

	//log(a-1)
	m = Must(Sub(x, onef64))
	lp = Must(Log(m))
	//TODO: surely there is a better way to test?
	if lp.children[0] == x {
		t.Error("Oops.")
	}
}

func TestExpStabilization(t *testing.T) {
	g := NewGraph()

	x := NewVector(g, Float64, WithName("x"), WithShape(2))
	e := Must(Exp(x))
	s := Must(Sub(e, onef64))

	if s.children[0] != x {
		t.Error("oops")
	}

	if euo, ok := s.op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != expm1OpType) {
		t.Error("oops")
	}

	if t.Failed() {
		ioutil.WriteFile("exp(a)-1.dot", []byte(s.ToDot()), 0644)
	}
}

func TestLogSigmoidStabilization(t *testing.T) {
	g := NewGraph()

	stabilization = true
	x := NewVector(g, Float64, WithName("x"), WithShape(2))
	y := Must(Sigmoid(x))
	WithName("y")(y)
	logY := Must(Log(y))
	WithName("log(sigmoid(x))")(logY)

	if euo, ok := logY.op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != negOpType) {
		t.Error("Oops")
	}

	if euo, ok := logY.children[0].op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != softplusOpType) {
		t.Error("Oops2")
	}

	if euo, ok := logY.children[0].children[0].op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != negOpType) {
		t.Error("Oops3")
	}

	if logY.children[0].children[0].children[0] != x {
		t.Errorf("Oops4: %v", logY.children[0].children[0].children[0].Name())
	}

	if t.Failed() {
		ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
		ioutil.WriteFile("logY.dot", []byte(logY.ToDot()), 0644)
	}
}
