package encoding

var groupIndex int

// NewGroup creates a new group with a generated ID
func NewGroup(name string) Group {
	g := Group{
		ID:   groupIndex,
		Name: name,
	}
	groupIndex++
	return g
}

// Group represent a cluster of elements
type Group struct {
	ID        int
	IsPrimary bool
	Name      string
}

// Grouper is any object that can claim itself as being part of a group
type Grouper interface {
	Groups() Groups
}

// Groups is a bag of groups
type Groups []Group

// Upsert the GroupID in the groups
func (g *Groups) Upsert(grp Group) {
	for i := 0; i < len(*g); i++ {
		if (*g)[i].ID == grp.ID {
			return
		}
	}
	*g = append(*g, grp)
}

// Have returns true if GroupID is in groups
func (g *Groups) Have(grp Group) bool {
	for i := 0; i < len(*g); i++ {
		if (*g)[i].ID == grp.ID {
			return true
		}
	}
	return false
}
