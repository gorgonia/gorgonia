package exprgraph

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/exprgraph/internal/uid"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	stdeng "gorgonia.org/tensor/engines"
)

type testGraphFields struct {
	Engine  tensor.Engine
	nodes   map[int64]Node
	from    map[int64][]int64
	to      map[int64][]int64
	self    float64
	absent  float64
	nodeIDs uid.Set
}

func graphFromFields(fields testGraphFields) *Graph {
	nodes := fields.nodes
	if nodes == nil {
		nodes = make(map[int64]Node)
	}
	from := fields.from
	if from == nil {
		from = make(map[int64][]int64)
	}
	to := fields.to
	if to == nil {
		to = make(map[int64][]int64)
	}
	g := &Graph{
		Engine:  fields.Engine,
		nodes:   nodes,
		from:    from,
		to:      to,
		self:    fields.self,
		absent:  fields.absent,
		nodeIDs: fields.nodeIDs,
	}
	return g
}

// simpleTestGraph returns the partial graph representation
// of the expression
//
//	c = a + b
//
// "partial" because the `Op` in the node is nil.
func simpleTestGraph() (f testGraphFields, a, b, c Node) {
	a = newSym(withID(0), withName("a"))
	b = newSym(withID(1), withName("b"))
	c = newSym(withID(2), withName("c"))

	return testGraphFields{
		nodes: map[int64]Node{
			0: a,
			1: b,
			2: c,
		},
		from: map[int64][]int64{
			2: {0, 1},
		},
		to: map[int64][]int64{
			0: {2},
			1: {2},
		},
	}, a, b, c

}

func TestNewGraph(t *testing.T) {
	type args struct {
		e tensor.Engine
	}
	tests := []struct {
		name string
		args args
		want *Graph
	}{
		{
			"simple",
			args{
				stdeng.Gen{},
			},
			&Graph{
				Engine:  stdeng.Gen{},
				nodes:   make(map[int64]Node),
				from:    make(map[int64][]int64),
				to:      make(map[int64][]int64),
				groups:  make(map[int64]encoding.Groups),
				self:    Self,
				absent:  Absent,
				nodeIDs: uid.NewSet(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewGraph(tt.args.e); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewGraph() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_Node(t *testing.T) {
	type args struct {
		id int64
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   graph.Node
	}{
		{
			"node exists",
			testGraphFields{
				nodes: map[int64]Node{
					1: newSym(),
				},
			},
			args{
				1,
			},
			newSym(),
		},
		{
			"node does not exists",
			testGraphFields{
				nodes: map[int64]Node{
					1: newSym(),
				},
			},
			args{
				2,
			},
			nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			got := g.Node(tt.args.id)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Node() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_AddChildren(t *testing.T) {
	type args struct {
		n        Node
		children []Node
	}
	tests := []struct {
		name    string
		fields  testGraphFields
		args    args
		wantErr bool
	}{
		{
			"main node is not found",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
				},
			},
			args{
				n: newSym(withID(1)),
			},
			true,
		},
		{
			"child node is not found",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
				},
			},
			args{
				n: newSym(),
				children: []Node{
					newSym(withID(1)),
				},
			},
			true,
		},
		{
			"child of itself",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
				},
			},
			args{
				n: newSym(withID(1)),
				children: []Node{
					newSym(),
				},
			},
			true,
		},
		{
			"all ok",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
					1: newSym(withID(1)),
				},
				from: make(map[int64][]int64),
				to:   make(map[int64][]int64),
			},
			args{
				n: newSym(),
				children: []Node{
					newSym(withID(1)),
				},
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if err := g.AddChildren(tt.args.n, tt.args.children...); (err != nil) != tt.wantErr {
				t.Errorf("Graph.AddChildren() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGraph_NameOf(t *testing.T) {
	sampleTensor := dense.New[float32](tensor.WithShape(1, 1))
	sampleNode := NewValue[float32]("test", sampleTensor)
	sampleDV := dual.New[float32](sampleTensor)
	sampleNodeLifted := &Value[float32, *dense.Dense[float32]]{
		desc:       desc{name: "test"},
		Basic:      sampleDV,
		beforeLift: sampleTensor,
	}

	type args struct {
		t Tensor
	}
	tests := []struct {
		name    string
		fields  testGraphFields
		args    args
		want    string
		wantErr bool
	}{
		{
			"nil",
			testGraphFields{},
			args{t: nil},
			"",
			true,
		},
		{
			"not found",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
				},
			},
			args{
				t: dense.New[float32](tensor.WithShape(1, 1)),
			},
			"",
			true,
		},
		{
			"tensor found",
			testGraphFields{
				nodes: map[int64]Node{
					0: NewValue[float32]("test", sampleTensor),
				},
			},
			args{
				t: sampleTensor,
			},
			"test",
			false,
		},
		{
			"tensor lifted found",
			testGraphFields{
				nodes: map[int64]Node{
					0: sampleNodeLifted,
				},
			},
			args{
				t: sampleTensor,
			},
			"test",
			false,
		},
		{
			"node found",
			testGraphFields{
				nodes: map[int64]Node{
					0: sampleNode,
				},
			},
			args{
				t: sampleNode,
			},
			"test",
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			got, err := g.NameOf(tt.args.t)
			if (err != nil) != tt.wantErr {
				t.Errorf("Graph.NameOf() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Graph.NameOf() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_IDOf(t *testing.T) {
	sampleTensor := dense.New[float32](tensor.WithShape(1, 1))
	sampleNode := NewValue[float32]("test", sampleTensor)
	sampleDV := dual.New[float32](sampleTensor)
	sampleNodeLifted := &Value[float32, *dense.Dense[float32]]{
		desc:       desc{name: "test"},
		Basic:      sampleDV,
		beforeLift: sampleTensor,
	}

	type args struct {
		t Tensor
	}
	tests := []struct {
		name    string
		fields  testGraphFields
		args    args
		want    NodeID
		wantErr bool
	}{
		{
			"not found",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
				},
			},
			args{
				t: dense.New[float32](tensor.WithShape(1, 1)),
			},
			NodeID(-1),
			true,
		},
		{
			"tensor found",
			testGraphFields{
				nodes: map[int64]Node{
					0: NewValue[float32]("test", sampleTensor),
				},
			},
			args{
				t: sampleTensor,
			},
			NodeID(0),
			false,
		},
		{
			"tensor lifted found",
			testGraphFields{
				nodes: map[int64]Node{
					0: sampleNodeLifted,
				},
			},
			args{
				t: sampleTensor,
			},
			NodeID(0),
			false,
		},
		{
			"node found",
			testGraphFields{
				nodes: map[int64]Node{
					0: sampleNode,
				},
			},
			args{
				t: sampleNode,
			},
			NodeID(0),
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			got, err := g.IDOf(tt.args.t)
			if (err != nil) != tt.wantErr {
				t.Errorf("Graph.IDOf() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Graph.IDOf() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_NodeOf(t *testing.T) {
	sampleTensor := dense.New[float32](tensor.WithShape(1, 1))
	sampleNode := NewValue[float32]("test", sampleTensor)
	sampleDV := dual.New[float32](sampleTensor)
	sampleNodeLifted := &Value[float32, *dense.Dense[float32]]{
		desc:       desc{name: "test"},
		Basic:      sampleDV,
		beforeLift: sampleTensor,
	}

	type args struct {
		t Tensor
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   Node
	}{
		{
			"not found",
			testGraphFields{},
			args{
				t: dense.New[float32](tensor.WithShape(1, 1)),
			},
			nil,
		},
		{
			"tensor found",
			testGraphFields{
				nodes: map[int64]Node{
					0: NewValue[float32]("test", sampleTensor),
				},
			},
			args{
				t: sampleTensor,
			},
			sampleNode,
		},
		{
			"node found",
			testGraphFields{
				nodes: map[int64]Node{
					0: sampleNode,
				},
			},
			args{
				t: sampleNode,
			},
			sampleNode,
		},
		{
			"lift",
			testGraphFields{
				nodes: map[int64]Node{
					0: sampleNodeLifted,
				},
			},
			args{
				t: sampleTensor,
			},
			sampleNodeLifted,
		},
		{
			"nil",
			testGraphFields{},
			args{t: nil},
			nil,
		},
		{
			"(Node)(nil)",
			testGraphFields{},
			args{t: (Node)(nil)},
			nil,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.NodeOf(tt.args.t); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.NodeOf() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_createEdge(t *testing.T) {
	type args struct {
		e graph.WeightedEdge
	}
	tests := []struct {
		name    string
		fields  testGraphFields
		args    args
		wantErr bool
	}{
		{
			"self node",
			testGraphFields{
				from: make(map[int64][]int64),
				to:   make(map[int64][]int64),
			},
			args{
				e: WeightedEdge{
					F: newSym(),
					T: newSym(),
				},
			},
			true,
		},
		// these tests are commented out because they are no longer pertinent (createEdge does not check for node existence.)
		/*
			{
				"from node not found",
				fields{
					nodes: map[int64]Node{
						1: {
							id: 1,
						},
					},
					from: make(map[int64][]int64),
					to:   make(map[int64][]int64),
				},
				args{
					e: WeightedEdge{
						F: &Node{
							id: 0,
						},
						T: &Node{
							id: 1,
						},
					},
				},
				true,
			},
			{
				"to node not found",
				fields{
					nodes: map[int64]Node{
						0: {
							id: 0,
						},
					},
					from: make(map[int64][]int64),
					to:   make(map[int64][]int64),
				},
				args{
					e: WeightedEdge{
						F: &Node{
							id: 0,
						},
						T: &Node{
							id: 1,
						},
					},
				},
				true,
			},
		*/
		{
			"ok, overriding existing links",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
					1: newSym(withID(1)),
				},
				from: make(map[int64][]int64),
				to:   make(map[int64][]int64),
			},
			args{
				e: WeightedEdge{
					F: newSym(),
					T: newSym(withID(1)),
				},
			},
			false,
		},
		{
			"ok, new fresh link",
			testGraphFields{
				nodes: map[int64]Node{
					0: newSym(),
					1: newSym(withID(1)),
				},
				from: make(map[int64][]int64),
				to:   make(map[int64][]int64),
			},
			args{
				e: WeightedEdge{
					F: newSym(),
					T: newSym(withID(1)),
				},
			},
			false,
		},
		//	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if err := g.createEdge(tt.args.e.From().(Node), tt.args.e.To().(Node)); (err != nil) != tt.wantErr {
				t.Errorf("Graph.setWeightedEdge() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGraph_Graph(t *testing.T) {
	tests := []struct {
		name   string
		fields testGraphFields
		want   *Graph
	}{
		{
			"simple",
			testGraphFields{},
			&Graph{
				nodes: map[int64]Node{},
				from:  map[int64][]int64{},
				to:    map[int64][]int64{},
			},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.Graph(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Graph() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_Nodes(t *testing.T) {
	first := newSym(withName("first"))
	tests := []struct {
		name   string
		fields testGraphFields
		want   graph.Nodes
	}{
		{
			"empty graph",
			testGraphFields{},
			graph.Empty,
		},
		{
			"one node",
			testGraphFields{
				nodes: map[int64]Node{
					0: first,
				},
			},
			&IterNodes{[]Node{first}, -1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.Nodes(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Nodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_From(t *testing.T) {
	type args struct {
		id int64
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   graph.Nodes
	}{
		{
			"empty graph",
			testGraphFields{},
			args{
				0,
			},
			graph.Empty,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.From(tt.args.id); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.From() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_HasEdgeBetween(t *testing.T) {
	type args struct {
		xid int64
		yid int64
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   bool
	}{
		{
			"link between x and y",
			testGraphFields{
				from: map[int64][]int64{
					0: {1},
				},
				to: map[int64][]int64{
					1: {0},
				},
			},
			args{
				0,
				1,
			},
			true,
		},
		{
			"link between y and x",
			testGraphFields{
				from: map[int64][]int64{
					0: {1},
				},
				to: map[int64][]int64{
					1: {0},
				},
			},
			args{
				1,
				0,
			},
			true,
		},
		{
			"no link between y and x",
			testGraphFields{
				from: map[int64][]int64{
					0: {1},
				},
				to: map[int64][]int64{
					1: {0},
				},
			},
			args{
				1,
				2,
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.HasEdgeBetween(tt.args.xid, tt.args.yid); got != tt.want {
				t.Errorf("Graph.HasEdgeBetween() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_HasEdgeFromTo(t *testing.T) {
	type args struct {
		uid int64
		vid int64
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   bool
	}{
		{
			"link between x and y",
			testGraphFields{
				from: map[int64][]int64{
					0: {1},
				},
				to: map[int64][]int64{
					1: {0},
				},
			},
			args{
				0,
				1,
			},
			true,
		},
		{
			"link between y and x",
			testGraphFields{
				from: map[int64][]int64{
					0: {1},
				},
				to: map[int64][]int64{
					1: {0},
				},
			},
			args{
				1,
				0,
			},
			false,
		},
		{
			"no link between y and x",
			testGraphFields{
				from: map[int64][]int64{
					// 0: {
					// 	1: &WeightedEdge{},
					// },
				},
			},
			args{
				1,
				2,
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.HasEdgeFromTo(tt.args.uid, tt.args.vid); got != tt.want {
				t.Errorf("Graph.HasEdgeFromTo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_AddNode(t *testing.T) {
	type args struct {
		n Node
	}
	tests := []struct {
		name    string
		fields  testGraphFields
		args    args
		wantErr bool
	}{
		{
			"invalid node id",
			testGraphFields{},
			args{newSym(withID(-1))},
			true,
		},
		{
			"node collision",
			testGraphFields{
				nodes: map[int64]Node{
					MinNodeID + 1: newSym(),
				},
			},
			args{newSym(withID(MinNodeID + 1))},
			true,
		},
		{
			"node ok",
			testGraphFields{
				nodes: map[int64]Node{
					MinNodeID + 1: newSym(),
				},
				nodeIDs: uid.NewSet(),
			},
			args{newVal(withID(MinNodeID + 2))},
			false,
		},
		{
			"node lifter ok",
			testGraphFields{
				nodes: map[int64]Node{
					MinNodeID + 1: newSym(),
				},
				nodeIDs: uid.NewSet(),
			},
			args{NewValue[float32]("test", dense.New[float32](dense.WithEngine(&dummyLifter{}), dense.WithShape(1, 1)))},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if err := g.AddNode(tt.args.n); (err != nil) != tt.wantErr {
				t.Errorf("Graph.AddNode() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

var _ Lifter = &dummyLifter{}

type dummyLifter struct {
	stdeng.Gen
}

func (*dummyLifter) Lift(t Tensor) Tensor {
	panic("NYI")
}

func TestGraph_Edge(t *testing.T) {
	type args struct {
		uid int64
		vid int64
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   graph.Edge
	}{
		{
			"simple",
			testGraphFields{
				from: map[int64][]int64{
					0: {1},
				},
				to: map[int64][]int64{
					1: {0},
				},
			},
			args{
				0, 1,
			},
			&WeightedEdge{
				F: (Node)(nil),
				T: (Node)(nil),
			},
		},
		{
			"no link",
			testGraphFields{
				from: map[int64][]int64{
					0: {
						//1: &WeightedEdge{},
					},
				},
			},
			args{
				0, 2,
			},
			nil,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.Edge(tt.args.uid, tt.args.vid); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Edge() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestGraph_To(t *testing.T) {
	simple, a, b, c := simpleTestGraph()
	type args struct {
		id int64
	}
	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   graph.Nodes
	}{
		{
			"nil graph",
			testGraphFields{
				to: map[int64][]int64{},
			},
			args{0},
			graph.Empty,
		},
		{
			"simple, to a",
			simple,
			args{a.ID()},
			&IterNodes{ns: []Node{c}, i: -1},
		},
		{
			"simple, to b",
			simple,
			args{b.ID()},
			&IterNodes{ns: []Node{c}, i: -1},
		},
		{
			"simple, to c",
			simple,
			args{c.ID()},
			graph.Empty,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.To(tt.args.id); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.To() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_Roots(t *testing.T) {
	dnode := newSym(withID(3), withName("d"))
	simple, anode, bnode, cnode := simpleTestGraph()

	tests := []struct {
		name    string
		fields  testGraphFields
		want    []Node
		altWant []Node // alternative results due to random ordering when iterating thru map
	}{
		{
			"nil graph",
			testGraphFields{},
			nil,
			nil,
		},
		{
			"simple",
			simple,
			[]Node{cnode},
			[]Node{cnode},
		},
		{
			"simple2",
			testGraphFields{
				nodes: map[int64]Node{
					0: anode,
					1: bnode,
					2: cnode,
					3: dnode,
				},
				from: map[int64][]int64{
					2: {0, 1},
				},
				to: map[int64][]int64{
					0: {2},
					1: {2},
				},
			},
			[]Node{cnode, dnode},
			[]Node{dnode, cnode},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.Roots(); !(reflect.DeepEqual(got, tt.want) || reflect.DeepEqual(got, tt.altWant)) {
				t.Errorf("Graph.Roots = %v. Want %v", got, tt.want)
			}
		})
	}

}

func TestGraph_ChildrenOf(t *testing.T) {
	simple, anode, bnode, cnode := simpleTestGraph()

	type args struct {
		node Nodelike
	}

	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   NodeIDs
	}{
		{
			"nil graph, want: nodeID(100)",
			testGraphFields{
				from: map[int64][]int64{},
				to:   map[int64][]int64{},
			},
			args{NodeID(100)},
			nil,
		},
		{
			"nil graph, want: nil",
			testGraphFields{},
			args{nil},
			nil,
		},
		{
			"nil graph, want: (Node)(nil)",
			testGraphFields{},
			args{(Node)(nil)},
			nil,
		},
		{
			"c = a + b, want: c (as NodeID)",
			simple,
			args{NodeID(2)},
			NodeIDs{0, 1},
		},
		{
			"c = a + b, want: c (as Node)",
			simple,
			args{cnode},
			NodeIDs{0, 1},
		},
		{
			"c = a + b, want: a",
			simple,
			args{anode},
			nil,
		},
		{
			"c = a + b, want: b",
			simple,
			args{bnode},
			nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.ChildrenOf(tt.args.node); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.ChildrenOf = %v. Want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_ParentsOf(t *testing.T) {
	simple, anode, bnode, cnode := simpleTestGraph()
	type args struct {
		node Nodelike
	}

	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   NodeIDs
	}{
		{
			"nil graph, want: nodeID(100)",
			testGraphFields{
				from: map[int64][]int64{},
				to:   map[int64][]int64{},
			},
			args{NodeID(100)},
			nil,
		},
		{
			"nil graph, want: nil",
			testGraphFields{},
			args{nil},
			nil,
		},
		{
			"nil graph, want: (Node)(nil)",
			testGraphFields{},
			args{(Node)(nil)},
			nil,
		},
		{
			"c = a + b, want: a (as NodeID)",
			simple,
			args{anode.NodeID()},
			NodeIDs{2},
		},
		{
			"c = a + b, want: b (as Node)",
			simple,
			args{bnode},
			NodeIDs{2},
		},

		{
			"c = a + b, want: c (as Node)",
			simple,
			args{cnode},
			nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.ParentsOf(tt.args.node); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.ParentsOf = %v. Want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_ParentsOfAsNodes(t *testing.T) {
	simple, anode, bnode, cnode := simpleTestGraph()
	type args struct {
		node Nodelike
	}

	tests := []struct {
		name   string
		fields testGraphFields
		args   args
		want   []Node
	}{
		{
			"nil graph, want: nodeID(100)",
			testGraphFields{
				from: map[int64][]int64{},
				to:   map[int64][]int64{},
			},
			args{NodeID(100)},
			nil,
		},
		{
			"nil graph, want: nil",
			testGraphFields{},
			args{nil},
			nil,
		},
		{
			"nil graph, want: (Node)(nil)",
			testGraphFields{},
			args{(Node)(nil)},
			nil,
		},
		{
			"c = a + b, want: a (as NodeID)",
			simple,
			args{anode.NodeID()},
			[]Node{cnode},
		},
		{
			"c = a + b, want: b (as Node)",
			simple,
			args{bnode},
			[]Node{cnode},
		},

		{
			"c = a + b, want: c (as Node)",
			simple,
			args{cnode},
			nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := graphFromFields(tt.fields)
			if got := g.ParentsOfAsNodes(tt.args.node); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.ParentsOf = %v. Want %v", got, tt.want)
			}
		})
	}
}
