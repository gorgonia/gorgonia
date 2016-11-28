package gorgonia

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
)

func ssBinOpTest(t *testing.T, op ʘBinaryOperatorType, dt Dtype) (err error) {
	assert := assert.New(t)
	var randX, randY interface{}
	switch dt {
	case Float64:
		randX = rand.ExpFloat64()
		randY = rand.ExpFloat64()
	case Float32:
		randX = float32(rand.ExpFloat64())
		randY = float32(rand.ExpFloat64())
	default:
		return errors.Errorf("op Test not yet implemented for %v ", op, dt)
	}

	binOp := newEBOByType(op, dt, dt)
	t.Logf("%v %v %v", randX, op, randY)

	var g, g2 *ExprGraph
	var x, y, z *Node
	var a, b, c *Node
	g = NewGraph()
	x = NewScalar(g, dt, WithName("x"))
	y = NewScalar(g, dt, WithName("y"))
	if z, err = applyOp(binOp, x, y); err != nil {
		return err
	}

	g2 = NewGraph()
	a = NewScalar(g2, dt, WithName("a"))
	b = NewScalar(g2, dt, WithName("b"))
	if c, err = applyOp(binOp, a, b); err != nil {
		return err
	}

	// var grads Nodes
	var m1 VM
	if op.isArith() {
		if _, err = Grad(c, a, b); err != nil {
			return err
		}
		m1 = NewLispMachine(g)
	} else {
		m1 = NewLispMachine(g, ExecuteFwdOnly())
	}

	prog, locMap, err := Compile(g2)
	if err != nil {
		return err
	}

	m2 := NewTapeMachine(prog, locMap)

	Let(x, randX)
	Let(y, randY)
	if err = m1.RunAll(); err != nil {
		return
	}

	Let(a, randX)
	Let(b, randY)
	if err = m2.RunAll(); err != nil {
		return
	}

	var xG, aG, yG, bG, zG, cG Value
	if op.isArith() {
		if xG, err = x.Grad(); err != nil {
			return
		}
		if yG, err = y.Grad(); err != nil {
			return
		}
		if aG, err = a.Grad(); err != nil {
			return
		}
		if bG, err = b.Grad(); err != nil {
			return
		}

		if zG, err = z.Grad(); err != nil {
			return
		}
		if cG, err = c.Grad(); err != nil {
			return
		}
		assert.True(ValueEq(xG, aG), "Test Diff of %v. xG != aG. Got %v and %v", op, xG, aG)
		assert.True(ValueEq(yG, bG), "Test Diff of %v. yG != bG. Got %v and %v", op, yG, bG)
		assert.True(ValueEq(zG, cG), "Test Diff of %v. zG != cG. Got %v and %v", op, zG, cG)
	}

	assert.True(ValueEq(x.Value(), a.Value()), "Test op %v. Values are different: x: %v, a %v", op, x.Value(), a.Value())
	assert.True(ValueEq(y.Value(), b.Value()), "Test op %v. Values are different: y: %v, b %v", op, y.Value(), b.Value())
	assert.True(ValueEq(z.Value(), c.Value()), "Test op %v. Values are different: z: %v, c %v", op, z.Value(), c.Value())

	return nil
}

func ttBinOpTest(t *testing.T, op ʘBinaryOperatorType, dt Dtype) (err error) {
	assert := assert.New(t)
	var x, y, z, a, b, c, cost *Node
	var g, g2 *ExprGraph

	var randX, randY interface{}
	switch dt {
	case Float32:
		randX = []float32{1, 2, 3, 4}
		randY = []float32{2, 2, 2, 2}
	case Float64:
		randX = []float64{1, 2, 3, 4}
		randY = []float64{2, 2, 2, 2}
	}

	// randX := Gaussian(0, 1)(dt, 2, 2)
	// randY := Gaussian(0, 1)(dt, 2, 2)

	xV := tensor.New(dtypeToTensorDtype(dt), tensor.WithShape(2, 2), tensor.WithBacking(randX))
	yV := tensor.New(dtypeToTensorDtype(dt), tensor.WithShape(2, 2), tensor.WithBacking(randY))

	g = NewGraph()
	g2 = NewGraph()
	x = NewMatrix(g, dt, WithName("x"), WithShape(2, 2))
	y = NewMatrix(g, dt, WithName("y"), WithShape(2, 2))
	a = NewMatrix(g2, dt, WithName("a"), WithShape(2, 2))
	b = NewMatrix(g2, dt, WithName("b"), WithShape(2, 2))

	binOp := newEBOByType(op, x.t, y.t)
	if z, err = applyOp(binOp, x, y); err != nil {
		return err
	}
	if c, err = applyOp(binOp, a, b); err != nil {
		return err
	}

	var m1 VM
	if op.isArith() {
		if _, err = Sum(z); err != nil {
			return err
		}
		if cost, err = Sum(c); err != nil {
			return err
		}

		if _, err = Grad(cost, a, b); err != nil {
			return err
		}
		m1 = NewLispMachine(g)
	} else {
		m1 = NewLispMachine(g, ExecuteFwdOnly())
	}

	prog, locMap, err := Compile(g2)
	if err != nil {
		return err
	}

	m2 := NewTapeMachine(prog, locMap, TraceExec())
	// m2 := NewTapeMachine(prog, locMap, TraceExec(), WithLogger(logger), WithWatchlist())

	Let(x, xV)
	Let(y, yV)
	if err = m1.RunAll(); err != nil {
		return
	}

	Let(a, xV)
	Let(b, yV)
	if err = m2.RunAll(); err != nil {
		return
	}

	var xG, aG, yG, bG, zG, cG Value
	if op.isArith() {
		if xG, err = x.Grad(); err != nil {
			return
		}
		if yG, err = y.Grad(); err != nil {
			return
		}
		if aG, err = a.Grad(); err != nil {
			return
		}
		if bG, err = b.Grad(); err != nil {
			return
		}

		if zG, err = z.Grad(); err != nil {
			return
		}
		if cG, err = c.Grad(); err != nil {
			return
		}
		assert.True(ValueEq(xG, aG), "Test Diff of %v. xG != aG. Got %+v \nand %+v", op, xG, aG)
		assert.True(ValueEq(yG, bG), "Test Diff of %v. yG != bG. Got %+v \nand %+v", op, yG, bG)
		assert.True(ValueEq(zG, cG), "Test Diff of %v. zG != cG. Got %+v \nand %+v", op, zG, cG)
	}

	assert.True(ValueEq(x.Value(), a.Value()), "Test op %v. Values are different: x: %+v\n a %+v", op, x.Value(), a.Value())
	assert.True(ValueEq(y.Value(), b.Value()), "Test op %v. Values are different: y: %+v\n b %+v", op, y.Value(), b.Value())
	assert.True(ValueEq(z.Value(), c.Value()), "Test op %v. Values are different: z: %+v\n c %+v", op, z.Value(), c.Value())

	if t.Failed() {
		ioutil.WriteFile(fmt.Sprintf("Test_%v_tt.dot", op), []byte(g2.ToDot()), 0644)
	}

	return nil
}

func TestBinOps(t *testing.T) {
	for op := addOpType; op < maxʘBinaryOpType; op++ {
		err := ssBinOpTest(t, op, Float64)
		if err != nil {
			t.Errorf("Float64 version err: %v", err)
		}

		err = ssBinOpTest(t, op, Float32)
		if err != nil {
			t.Errorf("Float32 version err: %v", err)
		}

		err = ttBinOpTest(t, op, Float64)
		if err != nil {
			t.Errorf("ttBinOp Float64 version err %v", err)
		}

		err = ttBinOpTest(t, op, Float32)
		if err != nil {
			t.Errorf("ttBinOp Float64 version err %v", err)
		}
	}
}
