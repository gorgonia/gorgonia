package encoding

// Cluster represent a group of nodes that are similars
// It is used for the grapviz generation

var (
	// UndefinedCluster ...
	UndefinedCluster = NewGroup("UndefinedCluster")
	// ExprGraphCluster is the default cluster
	ExprGraphCluster = NewGroup("ExprGraphCluster ")
	// ConstantCluster is the group of nodes that represents constants
	ConstantCluster = NewGroup("Constants ")
	// InputCluster is the group of nodes that are expecting values
	InputCluster = NewGroup("Inputs ")
	// GradientCluster ...
	GradientCluster = NewGroup("Gradients ")
	// StrayCluster ...
	StrayCluster = NewGroup("Undifferentiated nodes ")
)
