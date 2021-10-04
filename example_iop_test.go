package gorgonia_test

import (
	"fmt"
	"hash"
	"hash/fnv"
	"io/ioutil"

	"github.com/chewxy/hm"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MyNewOp struct{}

func (op MyNewOp) Arity() int { return 2 }
func (op MyNewOp) Type() hm.Type {
	t := TensorType{Dims: 4, Of: hm.TypeVariable('a')}
	return hm.NewFnType(t, t, t)
}
func (op MyNewOp) InferShape(ns ...DimSizer) (tensor.Shape, error) {
	return ns[0].(tensor.Shape).Clone(), nil
}

func (op MyNewOp) Do(values ...Value) (retVal Value, err error) {
	in1 := values[0]
	in2 := values[1]
	out, err := CloneValue(in1)
	if err != nil {
		return nil, err
	}
	return op.UsePreallocDo(out, in1, in2)
}

func (op MyNewOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	in1 := inputs[0]
	in2 := inputs[1]
	return tensor.Add(in1, in2, tensor.WithReuse(prealloc.(tensor.Tensor)))
}

func (op MyNewOp) ReturnsPtr() bool      { return true }
func (op MyNewOp) CallsExtern() bool     { return false }
func (op MyNewOp) OverwritesInput() int  { return -1 }
func (op MyNewOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "XXX") }
func (op MyNewOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}
func (op MyNewOp) String() string { return "XXX" }

func (op MyNewOp) DiffWRT(inputs int) []bool { return []bool{true, true, true} }

func (op MyNewOp) SymDiff(inputs Nodes, output *Node, grad *Node) (retVal Nodes, err error) {
	in1 := inputs[0]
	in2 := inputs[1]

	diffOp := MyNewDiffOp{op}
	g := in1.Graph()
	in2Diff := NewUniqueNode(WithType(in2.Type()), WithShape(in2.Shape().Clone()...), WithChildren(Nodes{in2}), In(g), WithOp(Iop{}))

	var in1Diff *Node
	if in1Diff, err = ApplyOp(diffOp, in1, in2, in2Diff); err != nil {
		return nil, err
	}
	return Nodes{in1Diff, in2Diff}, nil

}

type MyNewDiffOp struct{ MyNewOp }

func (op MyNewDiffOp) Arity() int { return 3 }
func (op MyNewDiffOp) Type() hm.Type {
	t := TensorType{Dims: 4, Of: hm.TypeVariable('a')}
	return hm.NewFnType(t, t, t, t)
}

func (op MyNewDiffOp) Do(values ...Value) (Value, error) {
	//in1 := values[0]
	in2 := values[1]
	in2Diff := values[2]

	retVal, err := CloneValue(in2)
	switch data := in2Diff.Data().(type) {
	case []float64:
		for i := range data {
			data[i] = 1000
		}
	}
	return retVal, err
}
func (op MyNewDiffOp) String() string { return "XXXDiff" }

func Example_iop() {
	g := NewGraph()
	x := NewTensor(g, tensor.Float64, 4, WithShape(4, 5, 6, 7), WithName("x"), WithInit(Ones()))
	y := NewTensor(g, tensor.Float64, 4, WithShape(4, 5, 6, 7), WithName("y"), WithInit(Zeroes()))
	z, err := ApplyOp(MyNewOp{}, x, y)
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := Sum(z)
	if err != nil {
		fmt.Println(err)
		return
	}
	_, err = Grad(s, x, y)
	if err != nil {
		fmt.Println(err)
		return
	}

	m := NewTapeMachine(g, BindDualValues(x, y, z), TraceExec())
	if err := m.RunAll(); err != nil {
		fmt.Println(err)
		return
	}

	yGrad, err := y.Grad()
	if err != nil {
		fmt.Println(err)
		return
	}

	all1000 := func(a []float64) bool {
		for i := range a {
			if a[i] != 1000 {
				return false
			}
		}
		return true
	}
	ioutil.WriteFile("xxx.dot", []byte(g.ToDot()), 0644)
	fmt.Printf("%v", all1000(yGrad.Data().([]float64)))

	// Output:
	// true
}
