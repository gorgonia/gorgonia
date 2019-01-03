package debugger

// Cluster represent a group of nodes that are similars
// It is used for the grapviz generation

const (
	// UndefinedCluster ...
	UndefinedCluster GroupID = iota
	// ExprGraphCluster is the default cluster
	ExprGraphCluster
	// ConstantCluster is the group of nodes that represents constants
	ConstantCluster
	// InputCluster is the group of nodes that are expecting values
	InputCluster
	// GradientCluster ...
	GradientCluster
	// StrayCluster ...
	StrayCluster
)
