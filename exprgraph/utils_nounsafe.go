// +build nounsafe

package exprgraph

func nodeIDs2IDs(a []NodeID) []int64 {
	retVal := make([]int64, 0, len(a))
	for _, n := range a {
		retVal = append(retVal, int64(n))
	}
	return retVal
}

func ids2NodeIDs(a []int64) []NodeID {
	retVal := make([]NodeID, 0, len(a))
	for _, n := range a {
		retVal = append(retVal, NodeID(n))
	}
	return retVal
}
