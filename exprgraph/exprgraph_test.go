package exprgraph

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph/internal/uid"
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
				from:    make(map[int64]map[int64]graph.WeightedEdge),
				to:      make(map[int64]map[int64]graph.WeightedEdge),
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
		from    map[int64]map[int64]graph.WeightedEdge
		to      map[int64]map[int64]graph.WeightedEdge
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
		from    map[int64]map[int64]graph.WeightedEdge
		to      map[int64]map[int64]graph.WeightedEdge
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
				from: make(map[int64]map[int64]graph.WeightedEdge),
				to:   make(map[int64]map[int64]graph.WeightedEdge),
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
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64]map[int64]graph.WeightedEdge
		to      map[int64]map[int64]graph.WeightedEdge
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		t gorgonia.Tensor
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
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64]map[int64]graph.WeightedEdge
		to      map[int64]map[int64]graph.WeightedEdge
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		t gorgonia.Tensor
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
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64]map[int64]graph.WeightedEdge
		to      map[int64]map[int64]graph.WeightedEdge
		self    float64
		absent  float64
		nodeIDs uid.Set
	}
	type args struct {
		t gorgonia.Tensor
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

func TestGraph_setWeightedEdge(t *testing.T) {
	type fields struct {
		Engine  tensor.Engine
		nodes   map[int64]*Node
		from    map[int64]map[int64]graph.WeightedEdge
		to      map[int64]map[int64]graph.WeightedEdge
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
			fields{},
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
		{
			"from node not found",
			fields{
				nodes: map[int64]*Node{
					1: {
						id: 1,
					},
				},
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
				from: map[int64]map[int64]graph.WeightedEdge{
					0: make(map[int64]graph.WeightedEdge),
				},
				to: map[int64]map[int64]graph.WeightedEdge{
					1: make(map[int64]graph.WeightedEdge),
				},
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
				from: make(map[int64]map[int64]graph.WeightedEdge, 0),
				to:   make(map[int64]map[int64]graph.WeightedEdge, 0),
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
			if err := g.setWeightedEdge(tt.args.e); (err != nil) != tt.wantErr {
				t.Errorf("Graph.setWeightedEdge() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
