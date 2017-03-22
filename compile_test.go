package gorgonia

import "testing"

func TestCompile_CompileFn(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float32, WithName("x"))
	y := NewScalar(g, Float32, WithName("y"))
	xpy := Must(Add(x, y))
	xmy := Must(Mul(x, y))
	x2 := Must(Square(x))

	progAll, _, err := Compile(g)
	if err != nil {
		t.Fatal(err)
	}

	progAdd, _, err := CompileFunction(g, Nodes{x, y}, Nodes{xpy})
	if err != nil {
		t.Fatal(err)
	}

	progMul, _, err := CompileFunction(g, Nodes{x, y}, Nodes{xmy})
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err = CompileFunction(g, Nodes{x, y}, Nodes{x2}); err == nil {
		t.Error("expected an error when there is an unused node")
	}

	// properties based testing
	if len(progAll.sorted) <= len(progAdd.sorted) || len(progAll.sorted) <= len(progMul.sorted) {
		t.Error("progAll should have more nodes included than progAdd or progMul")
	}

	if len(progAll.instructions) <= len(progAdd.instructions) || len(progAll.instructions) <= len(progMul.instructions) {
		t.Error("progAll should have more instructions than either progAdd or progMul")
	}

	// really this is more checking of the subgraphing
	if !progAdd.sorted.Contains(x) {
		t.Error("Expected progAdd to contain x")
	}
	if !progAdd.sorted.Contains(y) {
		t.Error("Expected progAdd to contain y")
	}
	if !progAdd.sorted.Contains(xpy) {
		t.Error("Expected progAdd to contain xpy")
	}
	if progAdd.sorted.Contains(xmy) || progAdd.sorted.Contains(x2) {
		t.Error("Expected progAdd to not contain either x2 or xmy")
	}

	// same as above
	if !progMul.sorted.Contains(x) {
		t.Error("Expected progMul to contain x")
	}
	if !progMul.sorted.Contains(y) {
		t.Error("Expected progMul to contain y")
	}
	if !progMul.sorted.Contains(xmy) {
		t.Error("Expected progMul to contain xmy")
	}
	if progMul.sorted.Contains(xpy) || progMul.sorted.Contains(x2) {
		t.Error("Expected progMul to not contain either x2 or xpy")
	}

}
