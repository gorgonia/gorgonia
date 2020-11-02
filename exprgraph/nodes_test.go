package exprgraph

import "testing"

func Test_nodeIDs_Contains(t *testing.T) {
	type args struct {
		a NodeID
	}
	tests := []struct {
		name string
		ns   nodeIDs
		args args
		want bool
	}{
		{
			"contains",
			[]NodeID{0, 1, 2},
			args{
				a: NodeID(1),
			},
			true,
		},
		{
			"Does not contain",
			[]NodeID{0, 1, 2},
			args{
				a: NodeID(3),
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.ns.Contains(tt.args.a); got != tt.want {
				t.Errorf("nodeIDs.Contains() = %v, want %v", got, tt.want)
			}
		})
	}
}
