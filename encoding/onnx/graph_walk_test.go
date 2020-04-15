package gorgonnx

import (
	"fmt"
	"hash"
	"testing"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// this dummyOp will only allows the creation of Gorgonia's ExprGraph
type dummyOp struct {
	arity int
	typ   hm.Type
}

func newDummyOp() operator {
	return &dummyOp{}
}

type errNilGorgoniaNode struct {
	node *Node
}

func (n *errNilGorgoniaNode) Error() string {
	return fmt.Sprintf("child %v is nil", n.node)
}

func (d *dummyOp) apply(g *Graph, n *Node) error {
	var err error
	children := getOrderedChildren(g.g, n)
	gorgoniaChildren := make([]*gorgonia.Node, len(children))
	for i := 0; i < len(children); i++ {
		if children[i].gorgoniaNode == nil {
			return &errNilGorgoniaNode{children[i]}
		}
		gorgoniaChildren[i] = children[i].gorgoniaNode
	}
	n.gorgoniaNode, err = gorgonia.ApplyOp(d, gorgoniaChildren...)
	return err
}
func (d *dummyOp) init(o onnx.Operation) error {
	d.arity = o.Attributes["arity"].(int)
	d.typ = o.Attributes["type"].(hm.Type)
	return nil
}

func (d *dummyOp) Arity() int {
	return d.arity
}

func (d *dummyOp) Type() hm.Type {
	return d.typ
	//t := gorgonia.TensorType{Dims: 1, Of: hm.TypeVariable('a')}
	//return hm.NewFnType(t, t)
}

func (*dummyOp) InferShape(ns ...gorgonia.DimSizer) (tensor.Shape, error) {
	return ns[0].(tensor.Shape).Clone(), nil
}

func (*dummyOp) Do(...gorgonia.Value) (gorgonia.Value, error) {
	return nil, nil
}

func (*dummyOp) ReturnsPtr() bool {
	return true
}

func (*dummyOp) CallsExtern() bool {
	return false
}

func (*dummyOp) OverwritesInput() int {
	return -1
}

func (*dummyOp) WriteHash(h hash.Hash) {
}

func (*dummyOp) Hashcode() uint32 {
	return 0
}

func (*dummyOp) String() string {
	return ""
}

func TestPopulateExprgraph_simple(t *testing.T) {
	register("dummy", newDummyOp)
	tensorType := gorgonia.TensorType{Dims: 1, Of: hm.TypeVariable('a')}
	g := NewGraph()
	g.exprgraph = gorgonia.NewGraph()
	// create a simple graph:
	// output -> dummyOp1
	// output -> dummyOp2
	// dummmuOp1 -> dummyOp2
	// dummyOp2 -> input1
	// dummyOp2 -> input2
	// output and input are tensors
	outputN := g.NewNode()
	g.AddNode(outputN)
	dummyOp1N := g.NewNode()
	g.AddNode(dummyOp1N)
	dummyOp2N := g.NewNode()
	g.AddNode(dummyOp2N)
	input1N := g.NewNode()
	g.AddNode(input1N)
	input2N := g.NewNode()
	g.AddNode(input2N)
	g.SetWeightedEdge(g.NewWeightedEdge(outputN, dummyOp1N, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(dummyOp1N, dummyOp2N, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(dummyOp2N, input1N, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(dummyOp2N, input2N, 0))

	// Now set the tensors...
	err := input1N.(*Node).SetTensor(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(1)))
	if err != nil {
		t.Fatal(err)
	}

	err = input2N.(*Node).SetTensor(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(1)))
	if err != nil {
		t.Fatal(err)
	}

	// Now apply the operation
	err = g.ApplyOperation(onnx.Operation{
		Name: "dummy",
		Attributes: map[string]interface{}{
			"arity": int(1),
			"type":  hm.NewFnType(tensorType, tensorType),
		},
	}, dummyOp1N)
	if err != nil {
		t.Fatal(err)
	}
	err = g.ApplyOperation(onnx.Operation{
		Name: "dummy",
		Attributes: map[string]interface{}{
			"arity": int(2),
			"type":  hm.NewFnType(tensorType, tensorType, tensorType),
		},
	}, dummyOp2N)

	if err != nil {
		t.Fatal(err)
	}
	err = g.ApplyOperation(onnx.Operation{
		Name: "dummy",
		Attributes: map[string]interface{}{
			"arity": int(1),
			"type":  hm.NewFnType(tensorType, tensorType),
		},
	}, outputN)
	if err != nil {
		t.Fatal(err)
	}

	err = g.populateExprgraph()
	if err != nil {
		t.Fatal(err)
	}
	root := g.exprgraph.Roots()[0]
	it := g.exprgraph.From(root.ID())
	if it.Len() != 1 {
		t.Fatalf("level1: bad number of children, expecte %v, got %v ", 1, it.Len())
	}
	// dummyOp1
	it.Next()
	n := it.Node()
	it = g.exprgraph.From(n.ID())
	if it.Len() != 1 {
		t.Fatalf("level2: bad number of children, expecte %v, got %v ", 1, it.Len())
	}
	// dummyOp2
	it.Next()
	n = it.Node()
	it = g.exprgraph.From(n.ID())
	if it.Len() != 2 {
		t.Fatalf("level3: bad number of children, expecte %v, got %v ", 2, it.Len())
	}

}
func TestPopulateExprgraph_complex(t *testing.T) {
	register("dummy", newDummyOp)

	tensorType := gorgonia.TensorType{Dims: 1, Of: hm.TypeVariable('a')}
	//return hm.NewFnType(t, t)
	g := NewGraph()
	g.exprgraph = gorgonia.NewGraph()
	// create a simple graph:
	// output -> dummyOp1
	// output -> dummyOp2
	// dummmuOp1 -> dummyOp2
	// dummyOp2 -> input
	// output and input are tensors
	outputN := g.NewNode()
	g.AddNode(outputN)
	dummyOp1N := g.NewNode()
	g.AddNode(dummyOp1N)
	dummyOp2N := g.NewNode()
	g.AddNode(dummyOp2N)
	inputN := g.NewNode()
	g.AddNode(inputN)
	g.SetWeightedEdge(g.NewWeightedEdge(outputN, dummyOp1N, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(dummyOp1N, dummyOp2N, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(outputN, dummyOp2N, 1))
	g.SetWeightedEdge(g.NewWeightedEdge(dummyOp2N, inputN, 0))

	// Now set the tensors...
	err := inputN.(*Node).SetTensor(tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(1)))
	if err != nil {
		t.Fatal(err)
	}

	// Now apply the operation
	err = g.ApplyOperation(onnx.Operation{
		Name: "dummy",
		Attributes: map[string]interface{}{
			"arity": int(1),
			"type":  hm.NewFnType(tensorType, tensorType),
		},
	}, dummyOp1N)
	if err != nil {
		t.Fatal(err)
	}
	err = g.ApplyOperation(onnx.Operation{
		Name: "dummy",
		Attributes: map[string]interface{}{
			"arity": int(1),
			"type":  hm.NewFnType(tensorType, tensorType),
		},
	}, dummyOp2N)

	if err != nil {
		t.Fatal(err)
	}
	err = g.ApplyOperation(onnx.Operation{
		Name: "dummy",
		Attributes: map[string]interface{}{
			"arity": int(2),
			"type":  hm.NewFnType(tensorType, tensorType, tensorType),
		},
	}, outputN)
	if err != nil {
		t.Fatal(err)
	}

	err = g.populateExprgraph()
	if err != nil {
		t.Fatal(err)
	}
	root := g.exprgraph.Roots()[0]
	it := g.exprgraph.From(root.ID())
	if it.Len() != 2 {
		t.Fatalf("level1: bad number of children, expecte %v, got %v ", 2, it.Len())
	}
	// dummyOp1
	it.Next()
	n := it.Node()
	it = g.exprgraph.From(n.ID())
	if it.Len() != 1 {
		t.Fatalf("level2: bad number of children, expecte %v, got %v ", 1, it.Len())
	}
	// dummyOp2
	it.Next()
	n = it.Node()
	it = g.exprgraph.From(n.ID())
	if it.Len() != 1 {
		t.Fatalf("level3: bad number of children, expecte %v, got %v ", 1, it.Len())
	}

}
