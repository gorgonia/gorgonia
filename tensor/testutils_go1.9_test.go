// +build go1.9

package tensor

import _ "unsafe"

//go:linkname clz math/bits.LeadingZeros
func clz(a uint) int

//go:linkname clz64 math/bits.LeadingZeros64
func clz64(a uint64) int

//go:linkname clz32 math/bits.LeadingZeros32
func clz32(a uint32) int

//go:linkname clz16 math/bits.LeadingZeros16
func clz16(a uint16) int

//go:linkname clz8 math/bits.LeadingZeros8
func clz8(a uint8) int