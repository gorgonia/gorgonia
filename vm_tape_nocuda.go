// +build !cuda

package gorgonia

type modules struct{}
type contexts struct{}
type functions struct{}

func (m modules) HasFunc(name string) bool                  { return false }
func (m modules) Function(name string) (interface{}, error) { return nil, nil }

func finalizeTapeMachine(m *tapeMachine) {}

func (m *tapeMachine) init() {}
