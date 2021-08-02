package cuda

import "gorgonia.org/shapes"

func logicalSize(s shapes.Shape) int {
	if s.IsScalar() {
		return 1
	}
	return s.TotalSize()
}
