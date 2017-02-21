// +build !cuda

package gorgonia

func finalizeTapeMachine(m *tapeMachine) {}

// UseCudaFor is an option for *tapeMachine. This function is NO-OP unless the program is built with the `cuda` tag.
func UseCudaFor(ops ...string) VMOpt {
	return func(m VM) {}
}
