package cuda

//go:generate cudagen

var cudaStdLib map[string]string
var cudaStdFuncs map[string][]string

const (
	elemBinOpMod   = "elembinop"
	elemUnaryOpMod = "elemunaryop"
)
