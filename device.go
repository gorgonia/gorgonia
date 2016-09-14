package gorgonia

//go:generate stringer -type=Device
type Device byte

const (
	CPU Device = iota
	GPU
)
