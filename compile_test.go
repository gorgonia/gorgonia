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
