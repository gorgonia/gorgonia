package exprgraph

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/exprgraph/internal/uid"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

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
				&tensor.StdEng{},
			},
			&Graph{
				Engine:  &tensor.StdEng{},
				nodes:   make(map[int64]*Node),
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
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		id int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   graph.Node
	}{
		{
			"node exists",
			fields{
				nodes: map[int64]*Node{
					1: {},
				},
			},
			args{
				1,
			},
			&Node{},
		},
		{
			"node does not exists",
			fields{
				nodes: map[int64]*Node{
					1: {},
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
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			got := g.Node(tt.args.id)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Node() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_AddChildren(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		n        *Node
		children []*Node
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"main node is not found",
			fields{
				nodes: map[int64]*Node{
					0: {},
				},
			},
			args{
				n: &Node{
					id: 1,
				},
			},
			true,
		},
		{
			"child node is not found",
			fields{
				nodes: map[int64]*Node{
					0: {
						id: 0,
					},
				},
			},
			args{
				n: &Node{
					id: 0,
				},
				children: []*Node{
					{
						id: 1,
					},
				},
			},
			true,
		},
		{
			"child of itself",
			fields{
				nodes: map[int64]*Node{
					0: {
						id: 0,
					},
				},
			},
			args{
				n: &Node{
					id: 0,
				},
				children: []*Node{
					{
						id: 0,
					},
				},
			},
			true,
		},
		{
			"all ok",
			fields{
				nodes: map[int64]*Node{
					0: {
						id: 0,
					},
					1: {
						id: 1,
					},
				},
				from: make(map[int64][]int64),
				to:   make(map[int64][]int64),
			},
			args{
				n: &Node{
					id: 0,
				},
				children: []*Node{
					{
						id: 1,
					},
				},
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if err := g.AddChildren(tt.args.n, tt.args.children...); (err != nil) != tt.wantErr {
				t.Errorf("Graph.AddChildren() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGraph_NameOf(t *testing.T) {
	sampleTensor := tensor.NewDense(tensor.Float32, tensor.Shape{1, 1})
	sampleNode := &Node{
		name:   "test",
		Tensor: sampleTensor,
	}
	sampleDV := dual.New(sampleTensor)
	sampleNodeLifted := &Node{
		name:       "test",
		Tensor:     sampleDV,
		beforeLift: sampleTensor,
	}
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		t Tensor
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    string
		wantErr bool
	}{
		{
			"not found",
			fields{
				nodes: map[int64]*Node{
					0: {},
				},
			},
			args{
				t: tensor.NewDense(tensor.Float32, tensor.Shape{1, 1}),
			},
			"",
			true,
		},
		{
			"tensor found",
			fields{
				nodes: map[int64]*Node{
					0: {
						name:   "test",
						Tensor: sampleTensor,
					},
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
			fields{
				nodes: map[int64]*Node{
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
			fields{
				nodes: map[int64]*Node{
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
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
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
	sampleTensor := tensor.NewDense(tensor.Float32, tensor.Shape{1, 1})
	sampleNode := &Node{
		name:   "test",
		Tensor: sampleTensor,
	}
	sampleDV := dual.New(sampleTensor)
	sampleNodeLifted := &Node{
		name:       "test",
		Tensor:     sampleDV,
		beforeLift: sampleTensor,
	}
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		t Tensor
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    NodeID
		wantErr bool
	}{
		{
			"not found",
			fields{
				nodes: map[int64]*Node{
					0: {},
				},
			},
			args{
				t: tensor.NewDense(tensor.Float32, tensor.Shape{1, 1}),
			},
			NodeID(-1),
			true,
		},
		{
			"tensor found",
			fields{
				nodes: map[int64]*Node{
					0: {
						name:   "test",
						Tensor: sampleTensor,
					},
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
			fields{
				nodes: map[int64]*Node{
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
			fields{
				nodes: map[int64]*Node{
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
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
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
	sampleTensor := tensor.NewDense(tensor.Float32, tensor.Shape{1, 1})
	sampleNode := &Node{
		name:   "test",
		Tensor: sampleTensor,
	}
	sampleDV := dual.New(sampleTensor)
	sampleNodeLifted := &Node{
		name:       "test",
		Tensor:     sampleDV,
		beforeLift: sampleTensor,
	}
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		t Tensor
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *Node
	}{
		{
			"not found",
			fields{
				nodes: map[int64]*Node{
					0: {},
				},
			},
			args{
				t: tensor.NewDense(tensor.Float32, tensor.Shape{1, 1}),
			},
			nil,
		},
		{
			"tensor found",
			fields{
				nodes: map[int64]*Node{
					0: {
						name:   "test",
						Tensor: sampleTensor,
					},
				},
			},
			args{
				t: sampleTensor,
			},
			&Node{
				name:   "test",
				Tensor: sampleTensor,
			},
		},
		{
			"node found",
			fields{
				nodes: map[int64]*Node{
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
			fields{
				nodes: map[int64]*Node{
					0: sampleNodeLifted,
				},
			},
			args{
				t: sampleTensor,
			},
			sampleNodeLifted,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.NodeOf(tt.args.t); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.NodeOf() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_createEdge(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		e graph.WeightedEdge
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"self node",
			fields{
				from: make(map[int64][]int64),
				to:   make(map[int64][]int64),
			},
			args{
				e: WeightedEdge{
					F: &Node{
						id: 0,
					},
					T: &Node{
						id: 0,
					},
				},
			},
			true,
		},
		// these tests are commented out because they are no longer pertinent (createEdge does not check for node existence.)
		/*
			{
				"from node not found",
				fields{
					nodes: map[int64]*Node{
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
					nodes: map[int64]*Node{
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
			fields{
				nodes: map[int64]*Node{
					0: {
						id: 0,
					},
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
			false,
		},
		{
			"ok, new fresh link",
			fields{
				nodes: map[int64]*Node{
					0: {
						id: 0,
					},
					1: {
						id: 1,
					},
				},
				from: make(map[int64][]int64, 0),
				to:   make(map[int64][]int64, 0),
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
			false,
		},
		//	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if err := g.createEdge(tt.args.e.From().(*Node), tt.args.e.To().(*Node)); (err != nil) != tt.wantErr {
				t.Errorf("Graph.setWeightedEdge() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGraph_Graph(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	tests := []struct {
		name   string
		fields fields
		want   *Graph
	}{
		{
			"simple",
			fields{},
			&Graph{},
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.Graph(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Graph() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_Nodes(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	tests := []struct {
		name   string
		fields fields
		want   graph.Nodes
	}{
		{
			"empty graph",
			fields{},
			graph.Empty,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.Nodes(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Nodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_From(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		id int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   graph.Nodes
	}{
		{
			"empty graph",
			fields{},
			args{
				0,
			},
			graph.Empty,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.From(tt.args.id); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.From() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_HasEdgeBetween(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		xid int64
		yid int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
	}{
		{
			"link between x and y",
			fields{
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
			fields{
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
			fields{
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
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.HasEdgeBetween(tt.args.xid, tt.args.yid); got != tt.want {
				t.Errorf("Graph.HasEdgeBetween() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_HasEdgeFromTo(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		uid int64
		vid int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   bool
	}{
		{
			"link between x and y",
			fields{
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
			fields{
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
			fields{
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
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.HasEdgeFromTo(tt.args.uid, tt.args.vid); got != tt.want {
				t.Errorf("Graph.HasEdgeFromTo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGraph_AddNode(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		n *Node
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"invalid node id",
			fields{},
			args{
				&Node{
					id: -1,
				},
			},
			true,
		},
		{
			"node collision",
			fields{
				nodes: map[int64]*Node{
					MinNodeID + 1: {},
				},
			},
			args{
				&Node{
					id: MinNodeID + 1,
				},
			},
			true,
		},
		{
			"node ok",
			fields{
				nodes: map[int64]*Node{
					MinNodeID + 1: {},
				},
				nodeIDs: uid.NewSet(),
			},
			args{
				&Node{
					Tensor: tensor.NewDense(tensor.Float32, tensor.Shape{1, 1}),
					id:     MinNodeID + 2,
				},
			},
			false,
		},
		{
			"node lifter ok",
			fields{
				nodes: map[int64]*Node{
					MinNodeID + 1: {},
				},
				nodeIDs: uid.NewSet(),
			},
			args{
				&Node{
					Tensor: tensor.NewDense(tensor.Float32, tensor.Shape{1, 1},
						tensor.WithEngine(&dummyLifter{}),
					),
					id: MinNodeID + 2,
				},
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if err := g.AddNode(tt.args.n); (err != nil) != tt.wantErr {
				t.Errorf("Graph.AddNode() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

var _ Lifter = &dummyLifter{}

type dummyLifter struct {
	tensor.StdEng
}

func (*dummyLifter) Lift(t Tensor) Tensor {
	return t.(tensor.Tensor).Clone().(tensor.Tensor)
}

func TestGraph_Edge(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		uid int64
		vid int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   graph.Edge
	}{
		{
			"simple",
			fields{
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
				F: (*Node)(nil),
				T: (*Node)(nil),
			},
		},
		{
			"no link",
			fields{
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
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.Edge(tt.args.uid, tt.args.vid); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.Edge() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestGraph_To(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64][]int64
		to      map[int64][]int64
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		id int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   graph.Nodes
	}{
		{
			"nil graph",
			fields{
				to: map[int64][]int64{},
			},
			args{
				0,
			},
			graph.Empty,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				Engine:  tt.fields.Engine,
				nodes:   tt.fields.nodes,
				from:    tt.fields.from,
				to:      tt.fields.to,
				self:    tt.fields.self,
				absent:  tt.fields.absent,
				nodeIDs: tt.fields.nodeIDs,
			}
			if got := g.To(tt.args.id); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Graph.To() = %v, want %v", got, tt.want)
			}
		})
	}
}
