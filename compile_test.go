package gorgonia

import "testing"

func TestCompile_medium(t *testing.T) {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(20, 20), WithName("x"))
	y := NewMatrix(g, Float64, WithShape(20, 20), WithName("y"))
	xpy := Must(Add(x, y))
	xmy := Must(Sub(x, y))
	xpys := Must(Slice(xpy, S(0, 10)))
	xpys2 := Must(Square(xpys))
	xmy2 := Must(Square(xmy))

	var final Value
	set := Set(xmy2, xpy)
	read := Read(xmy2, &final)

	prog, _, err := Compile(g)
	if err != nil {
		t.Fatalf("error while compiling: %v", err)
	}
	t.Log(prog)

	// check flushes
	var frag fragment
	var onDev bool
	frag = prog.m[xpys]
	switch xpy.op.(type) {
	case CUDADoer:
		onDev = true
		if _, ok := frag[0].(flushInstr); !ok {
			t.Error("Expected the first instruction to be a flush instr")
		}
	case CLDoer:
	default:
		// nothing
	}

	frag = prog.m[xpys2]
	if xpys2.op.CallsExtern() {
		if _, ok := frag[0].(alloc); !ok {
			t.Error("Expect the first instruction to be an alloc")
		}
	}

	// leakage test
	if onDev {
		reg0 := register{device: Device(0), id: 0}
		reg1 := register{device: Device(0), id: 1}
		reg2 := register{device: Device(0), id: 2}

		if !prog.instructions.has(free{reg0}) {
			t.Error("Expected GPU(0)0 to be freed")
		}

		if !prog.instructions.has(free{reg1}) {
			t.Error("Expected GPU(0)1 to be freed")
		}

		if !prog.instructions.has(free{reg2}) {
			t.Error("Expected GPU(0)2 to be freed")
		}
	}

	// position tests
	if onDev {
		frag = prog.m[set]
		if _, ok := frag[len(frag)-1].(free); !ok {
			t.Error("Expected a `free` instruction after LET")
		}

		frag = prog.m[read]
		if _, ok := frag[len(frag)-2].(free); !ok {
			t.Error("Expected a `free` instruction after READ")
		}
	}
}

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
