package tensor

import (
	"encoding/binary"
	"hash/fnv"
)

// hashIntArray uses fnv to generate an int
func hashIntArray(in []int) int {
	tmp := make([]byte, 8*len(in))
	for i := 0; i < len(in); i++ {
		binary.LittleEndian.PutUint64(tmp[i*8:i*8+8], uint64(in[i]))
	}
	h := fnv.New64a()
	v, _ := h.Write(tmp)
	return v
}

// func hashIntArrayPair(in1, in2 []int) int {
// 	n := len(in1) + len(in2)
// 	tmp := make([]byte, 8*n)
// 	i := 0
// 	for ; i < len(in1); i++ {
// 		binary.LittleEndian.PutUint64(tmp[i*8:i*8+8], uint64(in1[i]))
// 	}
// 	for j := 0; j < len(in2); j++ {
// 		binary.LittleEndian.PutUint64(tmp[i*8:i*8+8], uint64(in2[j]))
// 		i++
// 	}
// 	h := fnv.New64a()
// 	v, _ := h.Write(tmp)
// 	return v
// }
