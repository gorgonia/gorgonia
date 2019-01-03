package debugger

// GroupID represent a cluster of elements
type GroupID int

// Grouper is any object that can claim itself as being part of a group
type Grouper interface {
	Group() GroupID
}
