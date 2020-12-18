package exprgraph

import "errors"

// ErrNotFoundInGraph is returned anytime an information extracted from the graph is not found.
var ErrNotFoundInGraph = errors.New("not found in graph")
