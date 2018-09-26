package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/graph/iterator"
	"gonum.org/v1/gonum/graph/topo"
)

func TestForwardDiffAnalysis(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"))
	y := NewScalar(g, Float64, WithName("y"))
	z := NewScalar(g, Float64, WithName("z"))

	res1 := Must(Log(Must(Mul(x, y))))

	sorted, err := topo.Sort(g)
	if err != nil {
		t.Error(err)
	}

	sortedNodes := graphNodeToNode(iterator.NewOrderedNodes(sorted))
	affectsOutput, err := forwardDiffAnalysis(Nodes{res1}, sortedNodes)
	if err != nil {
		t.Error(err)
	}

	t.Logf("%v", affectsOutput)
	if affectsOutput.Contains(z) {
		t.Error("It shouldn't contain res2 or z")
	}
}

func TestBackwardDiffAnalysis(t *testing.T) {
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"))
	y := NewScalar(g, Float64, WithName("y"))
	z := NewScalar(g, Float64, WithName("z"))

	res1 := Must(Log(Must(Mul(x, y))))
	res2 := Must(Log(Must(Mul(x, y)))) // yes it's a duplicate

	sorted, err := topo.Sort(g)
	if err != nil {
		t.Error(err)
	}

	sortedNodes := graphNodeToNode(iterator.NewOrderedNodes(sorted))
	affectedByOutput, err := backwardDiffAnalysis(Nodes{x, y}, sortedNodes)
	if err != nil {
		t.Error(err)
	}

	t.Logf("%v", affectedByOutput)

	if !affectedByOutput.Contains(res1) || !affectedByOutput.Contains(res2) {
		t.Error("Expected res1 and res2 to be affected by wrts")
	}

	if affectedByOutput.Contains(z) {
		t.Error("z shouldn't be in the list at all")
	}
}

func TestBackprop(t *testing.T) {
	assert := assert.New(t)
	gradOut := NewConstant(ones(Float64), WithName("GradOut"))

	t.Log("Simple backprop")
	g := NewGraph()
	x := NewVector(g, Float64, WithName("x"), WithShape(10)) // horizontal vector
	y := NewVector(g, Float64, WithName("y"), WithShape(10)) // horizontal vector

	res := Must(Mul(x, y))

	grad := g.AddNode(gradOut)
	inputs := Nodes{x, y}
	ret, err := Backpropagate(Nodes{res}, Nodes{grad}, inputs)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(Nodes{inputs[1], grad}, ret[0].children)
	assert.Equal(Nodes{inputs[0], grad}, ret[1].children)
	assert.Equal(mulOpType, ret[0].op.(elemBinOp).ʘBinaryOperator.binOpType())
	assert.Equal(mulOpType, ret[1].op.(elemBinOp).ʘBinaryOperator.binOpType())

	// reset
	t.Log("Progressively more complex")
	g = NewGraph()
	x = NewMatrix(g, Float64, WithName("x"), WithShape(1, 10))  // row vector
	w := NewMatrix(g, Float64, WithName("w"), WithShape(10, 1)) // col vector

	mul := Must(Mul(x, w))
	res = Must(Exp(mul))

	grad = g.AddNode(gradOut)
	inputs = Nodes{x, w}
	if ret, err = Backpropagate(Nodes{res}, Nodes{grad}, inputs); err != nil {
		t.Error(err)
	}

	// Notes:
	//
	// extra was created in the Backprop process

	extra := Must(Mul(res, onef64))
	dzdx_expectedPath := Nodes{ret[0], w, extra, res, mul, x, w, grad}
	dzdw_expectedPath := Nodes{ret[1], x, extra, res, mul, x, w, grad}

	assert.True(dzdx_expectedPath.Equals(ret[0].seqWalk()))
	assert.True(dzdw_expectedPath.Equals(ret[1].seqWalk()))

	/*
		ioutil.WriteFile("Test_Res.dot", []byte(res.ToDot()), 0644)
		for i, n := range ret {
			WithName(fmt.Sprintf("dz/d%s", inputs[i].Name()))(n)
			ioutil.WriteFile(fmt.Sprintf("Test_Grad_%d.dot", i), []byte(n.ToDot()), 0644)
		}
		ioutil.WriteFile("WholeGraph.dot", []byte(g.ToDot()), 0644)
	*/
}

// Compound ops (like expm1, log1p and sigmoid) have fairly complex diff results. Got bitten by log1p's diffExpr, so here's the test for them all
func TestCompoundOpDiff(t *testing.T) {
	g := NewGraph()

	saved := stabilization
	stabilization = true
	defer func() {
		stabilization = saved
	}()

	// log1p
	x := NewVector(g, Float64, WithName("x"), WithShape(2))
	p := Must(Add(x, onef64))
	lp := Must(Log(p))
	op := lp.op.(elemUnaryOp)
	diffs, err := op.SymDiff(Nodes{x}, lp, onef64)
	if err != nil {
		t.Error(err)
	}

	if len(diffs) != 1 {
		t.Fatal("Expected only one result")
	}

	diff := diffs[0]
	ebo, ok := diff.op.(elemBinOp)
	if !ok || ok && ebo.binOpType() != divOpType {
		t.Error("Expected an elemBinOp")
		t.Error("Expected divOp to be the result of differentiating log1p")
	}
	if diff.children[0].Hashcode() != onef64.Hashcode() {
		t.Errorf("Expected 1 as the numerator. Got %v instead", diff.children[0])
	}
	ebo, ok = diff.children[1].op.(elemBinOp)
	if !ok || ok && ebo.binOpType() != addOpType {
		t.Error("Expected child1 to be (+)")
	}

}
