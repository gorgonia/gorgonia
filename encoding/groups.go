package encoding

// GroupID represent a cluster of elements
type GroupID int

// Grouper is any object that can claim itself as being part of a group
type Grouper interface {
	Groups() Groups
}

// Groups is a bag of groups
type Groups []GroupID

// Upsert the GroupID in the groups
func (g *Groups) Upsert(id GroupID) {
	for i := 0; i < len(*g); i++ {
		if (*g)[i] == id {
			return
		}
	}
	*g = append(*g, id)
}

// Have retuns true if GroupID is in groups
func (g *Groups) Have(id GroupID) bool {
	for i := 0; i < len(*g); i++ {
		if (*g)[i] == id {
			return true
		}
	}
	return false
}
