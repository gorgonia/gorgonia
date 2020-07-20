package xvm

import (
	"context"
	"reflect"
	"testing"

	"gorgonia.org/gorgonia"
)

func TestMachine_runAllNodes(t *testing.T) {
	inputC1 := make(chan ioValue, 0)
	outputC1 := make(chan gorgonia.Value, 1)
	inputC2 := make(chan ioValue, 0)
	outputC2 := make(chan gorgonia.Value, 1)

	n1 := &node{
		op:          &sumF32{},
		inputValues: make([]gorgonia.Value, 2),
		outputC:     outputC1,
		inputC:      inputC1,
	}
	n2 := &node{
		op:          &sumF32{},
		inputValues: make([]gorgonia.Value, 2),
		outputC:     outputC2,
		inputC:      inputC2,
	}
	type fields struct {
		nodes   []*node
		pubsubs *pubsub
	}
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			"simple",
			fields{
				nodes: []*node{n1, n2},
			},
			args{
				context.Background(),
			},
			false,
		},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		forty := gorgonia.F32(40.0)
		fortyTwo := gorgonia.F32(42.0)
		two := gorgonia.F32(2.0)
		t.Run(tt.name, func(t *testing.T) {
			m := &Machine{
				nodes:   tt.fields.nodes,
				pubsubs: tt.fields.pubsubs,
			}
			go func() {
				inputC1 <- struct {
					pos int
					v   gorgonia.Value
				}{
					0,
					&forty,
				}
				inputC1 <- struct {
					pos int
					v   gorgonia.Value
				}{
					1,
					&two,
				}
				inputC2 <- struct {
					pos int
					v   gorgonia.Value
				}{
					0,
					&forty,
				}
				inputC2 <- struct {
					pos int
					v   gorgonia.Value
				}{
					1,
					&two,
				}
			}()
			if err := m.runAllNodes(tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("Machine.runAllNodes() error = %v, wantErr %v", err, tt.wantErr)
			}
			out1 := <-outputC1
			out2 := <-outputC2
			if !reflect.DeepEqual(out1.Data(), fortyTwo.Data()) {
				t.Errorf("out1: bad result, expected %v, got %v", fortyTwo, out1)
			}
			if !reflect.DeepEqual(out2.Data(), fortyTwo.Data()) {
				t.Errorf("out2: bad result, expected %v, got %v", fortyTwo, out2)
			}
		})
	}
}

func TestNewMachine(t *testing.T) {
	g := gorgonia.NewGraph()
	forty := gorgonia.F32(40.0)
	//fortyTwo := gorgonia.F32(42.0)
	two := gorgonia.F32(2.0)
	n1 := gorgonia.NewScalar(g, gorgonia.Float32, gorgonia.WithValue(&forty), gorgonia.WithName("n1"))
	n2 := gorgonia.NewScalar(g, gorgonia.Float32, gorgonia.WithValue(&two), gorgonia.WithName("n2"))

	added, err := gorgonia.Add(n1, n2)
	if err != nil {
		t.Fatal(err)
	}
	i1 := newInput(n1)
	i2 := newInput(n2)
	op := newOp(added, true)
	gg := gorgonia.NewGraph()
	c1 := gorgonia.NewConstant(&forty)
	ic1 := newInput(c1)
	ic1.id = 0
	gg.AddNode(c1)
	type args struct {
		g *gorgonia.ExprGraph
	}
	tests := []struct {
		name string
		args args
		want *Machine
	}{
		{
			"nil graph",
			args{nil},
			nil,
		},
		{
			"simple graph WIP",
			args{
				g,
			},
			&Machine{
				nodes: []*node{
					i1, i2, op,
				},
			},
		},
		{
			"constant (arity 0)",
			args{
				gg,
			},
			&Machine{
				nodes: []*node{
					ic1,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewMachine(tt.args.g)
			if got == nil && tt.want == nil {
				return
			}
			if got == nil && tt.want != nil ||
				got != nil && tt.want == nil {
				t.Fatalf("NewMachine() = %v, want %v", got, tt.want)
			}
			if tt.want.nodes == nil && got.nodes != nil ||
				tt.want.nodes != nil && got.nodes == nil {
				t.Fatalf("NewMachine(nodes) = %v, want %v", got, tt.want)
			}
			if len(got.nodes) != len(tt.want.nodes) {
				t.Fatalf("bad number of nodes, expecting %v, got %v", len(tt.want.nodes), len(got.nodes))
			}
			for i := 0; i < len(got.nodes); i++ {
				compareNodes(t, got.nodes[i], tt.want.nodes[i])
			}
			/*
				if tt.want.pubsubs == nil && got.pubsubs != nil ||
					tt.want.pubsubs != nil && got.pubsubs == nil {
					t.Fatalf("NewMachine(pubsubs) = %v, want %v", got, tt.want)
				}
				if !reflect.DeepEqual(got.pubsubs, tt.want.pubsubs) {
					t.Fatalf("bad pubsubs, expecting %v, got %v", tt.want.pubsubs, got.pubsubs)
				}
			*/
		})
	}
}

func Test_createHub(t *testing.T) {
	type args struct {
		ns []*node
		g  *gorgonia.ExprGraph
	}
	tests := []struct {
		name string
		args args
		want []*pubsub
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := createNetwork(tt.args.ns, tt.args.g); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("createHub() = %v, want %v", got, tt.want)
			}
		})
	}
}
