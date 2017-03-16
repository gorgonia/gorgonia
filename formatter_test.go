package gorgonia

import (
	"fmt"
	"testing"
)

func TestMapFormat(t *testing.T) {

	g := NewGraph()
	x := NewVector(g, Float64, WithName("x"))
	y := NewScalar(g, Float64, WithName("y"))

	m := make(map[*Node]string)
	m[x] = "x"
	m[y] = "y"

	s := fmt.Sprintf("%-#v", FmtNodeMap(m))
	expected0 := fmt.Sprintf("map[Node.ID]string {\n\t%x: x\n\t%x: y\n}", x.ID(), y.ID())
	expected1 := fmt.Sprintf("map[Node.ID]string {\n\t%x: y\n\t%x: x\n}", y.ID(), x.ID())
	if s != expected0 && s != expected1 {
		t.Errorf("Case 1 failed. Got \n%v", s)
	}

	m2 := make(map[*Node]*Node)
	m2[x] = x
	m2[y] = y

	// s = fmt.Sprintf("%+#v", FmtNodeMap(m2))
	// expected0 = fmt.Sprintf("map[Node.ID]*gorgonia.Node {\n\t%x: x :: Vector float64\n\t%x: y :: float64\n}", x.ID(), y.ID())
	// expected1 = fmt.Sprintf("map[Node.ID]*gorgonia.Node {\n\t%x: y :: float64\n\t%x: x :: Vector float64\n}", y.ID(), x.ID())
	// if s != expected0 && s != expected1 {
	// 	t.Errorf("Case 2 failed. Expected : %q. Got %q instead", expected0, s)
	// }

	s = fmt.Sprintf("%-#d", FmtNodeMap(m2))
	expected0 = fmt.Sprintf("map[Node.ID]*gorgonia.Node {\n\t%x: %x\n\t%x: %x\n}", x.ID(), x.ID(), y.ID(), y.ID())
	expected1 = fmt.Sprintf("map[Node.ID]*gorgonia.Node {\n\t%x: %x\n\t%x: %x\n}", y.ID(), y.ID(), x.ID(), x.ID())
	if s != expected0 && s != expected1 {
		t.Errorf("Case 3 failed")
	}

	m3 := make(map[*Node]Nodes)
	m3[x] = Nodes{x, y}
	s = fmt.Sprintf("%-v", FmtNodeMap(m3))
	expected0 = fmt.Sprintf("map[Node.Name]gorgonia.Nodes {\n\tx :: Vector float64: [x, y]\n}")
	if s != expected0 {
		t.Errorf("Case 4 failed. Expected : %q. Got %q instead", expected0, s)
	}

	/* TODO: COME BACK TO THIS

	s = fmt.Sprintf("%#-d", FmtNodeMap(m3))
	expected0 = fmt.Sprintf("map[Node.ID]gorgonia.Nodes {\n\t%x: [%x, %x]\n}", x.ID(), x.ID(), y.ID())
	if s != expected0 {
		t.Errorf("Case 5 failed. Got %q instead of %q", s, expected0)
	}
	*/
}
