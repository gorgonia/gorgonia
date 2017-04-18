// +build !cuda

package gorgonia

func (m *lispMachine) init() error {
	if err := m.prepGraph(); err != nil {
		return err
	}
}

func (m *lispMachine) execDevTrans(op devTrans, n *Node, children Nodes) (err error) { return nil }
