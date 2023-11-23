//go:build !nounsafe
// +build !nounsafe

package exprgraph

import "unsafe"

func nodeIDs2IDs(a []NodeID) []int64 { return *(*[]int64)(unsafe.Pointer(&a)) }

func ids2NodeIDs(a []int64) []NodeID { return *(*[]NodeID)(unsafe.Pointer(&a)) }
