package encoding

import "testing"

func TestGroups(t *testing.T) {
	var g Groups
	g.Upsert(1)
	if len(g) != 1 {
		t.Fail()
	}
	g.Upsert(1)
	if len(g) != 1 {
		t.Fail()
	}
	g.Upsert(2)
	if len(g) != 2 {
		t.Fail()
	}
	t.Log(g)
	if g[0] != 1 && g[1] != 2 {
		t.Fail()
	}
	if !g.Have(1) {
		t.Fail()
	}
	if !g.Have(2) {
		t.Fail()
	}
	if g.Have(3) {
		t.Fail()
	}

}
