package gorgonia

// BroadcastAdd  adds two nodes with the correct broadcasting
func BroadcastAdd(a, b *Node, leftPattern, rightPattern []byte) (*Node, error) {
	a2, b2, err := Broadcast(a, b, NewBroadcastPattern(leftPattern, rightPattern))
	if err != nil {
		return nil, err
	}
	return Add(a2, b2)
}

// BroadcastMul  multiplies (hadamardprod) two nodes with the correct broadcasting
func BroadcastMul(a, b *Node, leftPattern, rightPattern []byte) (*Node, error) {
	a2, b2, err := Broadcast(a, b, NewBroadcastPattern(leftPattern, rightPattern))
	if err != nil {
		return nil, err
	}
	return HadamardProd(a2, b2)
}
