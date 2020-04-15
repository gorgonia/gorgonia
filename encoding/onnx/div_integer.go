package gorgonnx

func floorDivInt(a, b int) int {
	if a%b == 0 {
		return a / b
	}

	div := a / b
	if a < 0 && b > 0 || a > 0 && b < 0 {
		return div - 1
	}
	return div
}
func ceilDivInt(a, b int) int {
	if a%b == 0 {
		return a / b
	}
	return floorDivInt(a, b) + 1
}
