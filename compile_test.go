package gorgonia

import (
	"io/ioutil"
	"testing"
)

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
	ioutil.WriteFile("compile_medium.dot", []byte(g.ToDot()), 0644)

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

	// free could be attached to either `read` or `set`
	var readFreeL, readFree2L, setFreeL, setFree2L bool
	frag = prog.m[read]
	switch xpy.op.(type) {
	case CUDADoer:
		if _, ok := frag[len(frag)-1].(free); ok {
			readFreeL = true
		}
		if len(frag) > 2 {
			if _, ok := frag[len(frag)-2].(free); ok {
				readFree2L = true
			}
		}
	case CLDoer:
	default:
	}

	frag = prog.m[set]
	switch xpy.op.(type) {
	case CUDADoer:
		if _, ok := frag[len(frag)-1].(free); ok {
			setFreeL = true
		}
		if len(frag) > 2 {
			if _, ok := frag[len(frag)-2].(free); ok {
				setFree2L = true
			}
		}
	case CLDoer:
	default:
	}

	switch {
	case !onDev:
	case onDev && readFree2L && readFreeL:
	case onDev && setFree2L && setFreeL:
	case onDev && setFreeL && readFreeL:
	default:
		t.Errorf("Expected free to be in either `set` or `read`. readFree2L: %t, readFreeL %t, setFree2L: %t, setFreeL: %t", readFree2L, readFreeL, setFree2L, setFreeL)
	}

	// t.Log(prog)
	// t.Log(prog.m[xpys])
	// t.Log(prog.m[xpys2])
	// t.Log(prog.m[xmy2])
	// t.Log(locMap)
	// t.Log(locMap[xpys2])
	// t.Log(locMap[xmy2])
}
