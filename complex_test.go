package gorgonia

import "testing"

func TestWeirdNetwork(t *testing.T) {
	const (
		embeddingDims = 50
		hiddenSize    = 200

		xs     = 64
		xFeats = 20

		ps     = 20
		pFeats = 10

		qs     = 50
		qFeats = 12

		outSize = 10
	)
	var err error

	g := NewGraph()
	var x *Node // NewVector(g, Float64, WithShape(xFeats*embeddingDims), WithName("x"), WithInit(Zeroes()))
	var p *Node
	var q *Node

	eX := NewMatrix(g, Float64, WithShape(xs, embeddingDims), WithName("x embeddings"), WithInit(GlorotU(1)))
	eP := NewMatrix(g, Float64, WithShape(ps, embeddingDims), WithName("p embeddings"), WithInit(GlorotU(1)))
	eQ := NewMatrix(g, Float64, WithShape(qs, embeddingDims), WithName("q embeddings"), WithInit(GlorotU(1)))
	w0X := NewMatrix(g, Float64, WithShape(hiddenSize, xFeats*embeddingDims), WithName("layer0 weights for x"), WithInit(GlorotU(1)))
	w0P := NewMatrix(g, Float64, WithShape(hiddenSize, pFeats*embeddingDims), WithName("layer0 weights for p"), WithInit(GlorotU(1)))
	w0Q := NewMatrix(g, Float64, WithShape(hiddenSize, qFeats*embeddingDims), WithName("layer0 weights for q"), WithInit(GlorotU(1)))
	b := NewVector(g, Float64, WithShape(hiddenSize), WithName("bias"), WithInit(Zeroes()))
	w1 := NewMatrix(g, Float64, WithShape(outSize, hiddenSize), WithName("layer 1"), WithInit(GlorotU(1)))

	model := Nodes{eX, eP, eQ, w0X, w0P, w0Q, b, w1}

	/* SET UP NEURAL NETWORK */

	slicesX := make(Nodes, xFeats)
	slicesP := make(Nodes, pFeats)
	slicesQ := make(Nodes, qFeats)

	for i := 0; i < xFeats; i++ {
		if slicesX[i], err = Slice(eX, S(i)); err != nil {
			t.Fatal(err)
		}
	}

	for i := 0; i < pFeats; i++ {
		if slicesP[i], err = Slice(eP, S(i)); err != nil {
			t.Fatal(err)
		}
	}

	for i := 0; i < qFeats; i++ {
		if slicesQ[i], err = Slice(eQ, S(i)); err != nil {
			t.Fatal(err)
		}
	}

	if x, err = Concat(0, slicesX...); err != nil {
		t.Fatal(err)
	}

	if p, err = Concat(0, slicesP...); err != nil {
		t.Fatal(err)
	}

	if q, err = Concat(0, slicesQ...); err != nil {
		t.Fatal(err)
	}

	var wx, wp, wq *Node
	if wx, err = Mul(w0X, x); err != nil {
		t.Fatal(err)
	}

	if wp, err = Mul(w0P, p); err != nil {
		t.Fatal(err)
	}

	if wq, err = Mul(w0Q, q); err != nil {
		t.Fatal(err)
	}

	// add all them layers
	var add0, add1, add2 *Node
	if add0, err = Add(wx, wp); err != nil {
		t.Fatal(err)
	}
	if add1, err = Add(add0, wq); err != nil {
		t.Fatal(err)
	}
	if add2, err = Add(add1, b); err != nil {
		t.Fatal(err)
	}

	// activate
	var act0 *Node
	if act0, err = Cube(add2); err != nil {
		t.Fatal(err)
	}

	// layer 1
	var layer1 *Node
	if layer1, err = Mul(w1, act0); err != nil {
		t.Fatal(err)
	}

	// activate
	var logProb *Node
	if logProb, err = SoftMax(layer1); err != nil {
		t.Fatal(err)
	}

	var cost *Node
	if cost, err = Slice(logProb, S(0)); err != nil { // dummy slice
		t.Fatal(err)
	}

	// backprop
	if _, err = Grad(cost, model...); err != nil {
		t.Fatal(err)
	}

	/* SET UP COMPLETE */

	m := NewTapeMachine(g, BindDualValues(model...))
	defer m.Close()

	// for debug purposes
	// prog, locMap, err := Compile(g)
	// log.Println(prog)

	// for i := 0; i < 104729; i++ {
	for i := 0; i < 2; i++ {
		if err = m.RunAll(); err != nil {
			t.Errorf("%d %v", i, err)
			break
		}

		m.Reset()
	}

}
