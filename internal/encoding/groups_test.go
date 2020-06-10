package encoding

import (
	"sort"
	"testing"
)

func TestGroups(t *testing.T) {
	var g Groups
	g1 := NewGroup("g1")
	g2 := NewGroup("g2")
	g = g.Upsert(g1)
	if len(g) != 1 {
		t.Fail()
	}
	g = g.Upsert(g1)
	if len(g) != 1 {
		t.Fail()
	}
	g = g.Upsert(g2)
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

func TestSorts(t *testing.T) {
	var gs Groups
	g1 := NewGroup("A")
	g2 := NewGroup("B")

	gs = gs.Upsert(g2)
	gs = gs.Upsert(g1)

	t.Logf("%v", gs)
	sort.Sort(gs)
	if gs[0] != g1 {
		t.Errorf("Groups: Expected the first element to be Group A. g1's ID %d", g1.ID)
	}
	if gs[1] != g2 {
		t.Errorf("Groups: Expected the second element to be Group B, g2's ID %d", g2.ID)
	}

	t.Logf("%v", gs)

	// reset

	gs = gs[:0]
	gs = gs.Upsert(g2)
	gs = gs.Upsert(g1)

	// check that resets work
	if gs[0] != g2 {
		t.Fatal("Must not proceed. Reset failed")
	}

	sort.Sort(ByName(gs))
	if gs[0] != g1 {
		t.Errorf("ByName: Expected the first element to be Group A. g1's name: %q", g1.Name)
	}
	if gs[1] != g2 {
		t.Errorf("ByName: Expected the second element to be Group B. g2's name: %q", g2.Name)
	}

}
