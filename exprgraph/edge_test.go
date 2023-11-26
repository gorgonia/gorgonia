package exprgraph

import (
	"testing"
)

func TestWeightedEdge(t *testing.T) {
	from := &desc{
		id: 0,
	}
	to := &desc{
		id: 1,
	}
	we := WeightedEdge{
		F: from,
		T: to,
		W: 0,
	}
	reverse := WeightedEdge{
		F: to,
		T: from,
		W: 0,
	}
	if we.From() != from {
		t.Fail()
	}
	if we.To() != to {
		t.Fail()
	}
	if we.Weight() != 0 {
		t.Fail()
	}
	if we.ReversedEdge() != reverse {
		t.Fail()
	}
}
