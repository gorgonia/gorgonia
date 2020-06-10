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
func (g Groups) Upsert(grp Group) Groups {
	for i := 0; i < len(g); i++ {
		if (g)[i].ID == grp.ID {
			return g
		}
	}
	return append(g, grp)
}

// Have returns true if GroupID is in groups
func (g Groups) Have(grp Group) bool {
	for i := 0; i < len(g); i++ {
		if (g)[i].ID == grp.ID {
			return true
		}
	}
	return false
}

/* Groups by default sort by the group ID */

// Len returns the length of a bag of groups
func (g Groups) Len() int { return len(g) }

// Less checks if an ID is less than or not
func (g Groups) Less(i, j int) bool { return g[i].ID < g[j].ID }

// Swap swaps the elements
func (g Groups) Swap(i, j int) { g[i], g[j] = g[j], g[i] }

// ByName is a sorting for a slice of groups, where the groups are sorted by name
type ByName []Group

func (g ByName) Len() int { return len(g) }

func (g ByName) Less(i, j int) bool { return g[i].Name < g[j].Name }

func (g ByName) Swap(i, j int) { g[i], g[j] = g[j], g[i] }
