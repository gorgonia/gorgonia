package encoding

import "testing"

func TestGroups(t *testing.T) {
	var g Groups
	g1 := NewGroup("g1")
	g2 := NewGroup("g2")
	g.Upsert(g1)
	if len(g) != 1 {
		t.Fail()
	}
	g.Upsert(g1)
	if len(g) != 1 {
		t.Fail()
	}
	g.Upsert(g2)
	if len(g) != 2 {
		t.Fail()
	}
	if g[0].Name != "g1" && g[1].Name != "g2" {
		t.Fail()
	}
	if !g.Have(g1) {
		t.Fail()
	}
	if !g.Have(g2) {
		t.Fail()
	}
	if g.Have(NewGroup("g2")) {
		t.Fail()
	}

}
