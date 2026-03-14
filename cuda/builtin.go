//go:build !darwin || !arm64
package cuda

//go:generate cudagen

const (
	elemBinOpMod   = "elembinop"
	elemUnaryOpMod = "elemunaryop"
)
