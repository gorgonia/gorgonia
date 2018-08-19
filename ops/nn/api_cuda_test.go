// +build cuda

package nnops

import (
	"io/ioutil"
	"log"
	"os"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDropout(t *testing.T) {
	g := G.NewGraph()
	x := G.NewMatrix(g, G.Float64, G.WithShape(2, 3), G.WithName("x"))
	do, _ := Dropout(x, 0.5)
	log.Printf("%v", do)
	ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)

}

/*
func TestBatchNorm_F64(t *testing.T) {
	g := G.NewGraph()
	x := G.NewTensor(g, G.Float64, 4, G.WithShape(5, 2, 3, 4), G.WithInit(G.Gaussian(0, 1)))
	y, op, err := BatchNorm(x, 0.9, 1e-5, true)
	if err != nil {
		t.Fatal(err)
	}

	var yVal G.Value
	G.Read(y, &yVal)

	cost, _ := G.Mean(y)

	if _, err := G.Grad(cost, x); err != nil {
		t.Fatal(err)
	}

	m := G.NewTapeMachine(g, G.BindDualValues(x), G.TraceExec())
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	m.Close()
	ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)

	shape := x.Shape()
	n, c, h, w := shape[0], shape[1], shape[2], shape[3]

	yVT := yVal.(*tensor.Dense)
	for j := 0; j < c; j++ {
		var sum, variance float64
		for i := 0; i < n; i++ {
			for k := 0; k < h; k++ {
				for l := 0; l < w; l++ {
					at, err := yVT.At(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					atf := at.(float64)
					sum += atf
					variance += atf * atf
				}
			}
		}
		sum /= float64(h * w * n)
		variance /= float64(h * w * n)

		if !dawson.ToleranceF64(sum, 0, 0.00001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF64(variance, 1, 0.0001) {
			t.Errorf("channel %d: Expected variance to be near 1. Got %v", j, variance)
		}
	}

	op.SetTesting()
	m = G.NewTapeMachine(g, G.BindDualValues(x))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	m.Close()
	yVT = yVal.(*tensor.Dense)
	for j := 0; j < c; j++ {
		var sum, variance float64
		for i := 0; i < n; i++ {
			for k := 0; k < h; k++ {
				for l := 0; l < w; l++ {
					at, err := yVT.At(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					atf := at.(float64)
					sum += atf
					variance += atf * atf
				}
			}
		}
		sum /= float64(h * w * n)
		variance /= float64(h * w * n)

		if !dawson.ToleranceF64(sum, 0, 0.00001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF64(variance, 0.9833, 0.0001) {
			t.Errorf("channel %d: Expected variance to be near 0.98. Got %v", j, variance)
		}
	}
}
*/

func TestDevBN(t *testing.T) {
	g := G.NewGraph()
	x := G.NewTensor(g, G.Float64, 4, G.WithShape(5, 2, 3, 4), G.WithInit(G.Gaussian(0, 1)), G.WithName("x"))
	y, _, _, op, err := BatchNorm(x, nil, nil, 0.9, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)

	cost, _ := G.Mean(y)

	if _, err := G.Grad(cost, x); err != nil {
		t.Fatal(err)
	}

	log.Printf("%v | %v", y, op)
	ioutil.WriteFile("bar.dot", []byte(g.ToDot()), 0644)
	prog, _, _ := G.Compile(g)
	log.Printf("%v", prog)
	logger := log.New(os.Stderr, "", 0)
	m := G.NewTapeMachine(g, G.BindDualValues(x), G.WithLogger(logger), G.WithWatchlist())
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	m.Close()
}

func TestScratch(t *testing.T) {
	g := G.NewGraph()
	ss := &scratchOp{tensor.Shape{1, 2, 3, 4}, tensor.Float64, "testScratch"}
	x := G.NewTensor(g, tensor.Float64, 4, G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithOp(ss))
	prog, _, _ := G.Compile(g)
	log.Printf("x: %v y: %v", x, y)
	log.Printf("%v", prog)
}
