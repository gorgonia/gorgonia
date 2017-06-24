package tensor

import "unsafe"

/*
GENERATED FILE. DO NOT EDIT
*/

/* bool */

func (h *header) bools() []bool      { return *(*[]bool)(unsafe.Pointer(h)) }
func (h *header) setB(i int, x bool) { h.bools()[i] = x }
func (h *header) getB(i int) bool    { return h.bools()[i] }

/* int */

func (h *header) ints() []int       { return *(*[]int)(unsafe.Pointer(h)) }
func (h *header) setI(i int, x int) { h.ints()[i] = x }
func (h *header) getI(i int) int    { return h.ints()[i] }

/* int8 */

func (h *header) int8s() []int8       { return *(*[]int8)(unsafe.Pointer(h)) }
func (h *header) setI8(i int, x int8) { h.int8s()[i] = x }
func (h *header) getI8(i int) int8    { return h.int8s()[i] }

/* int16 */

func (h *header) int16s() []int16       { return *(*[]int16)(unsafe.Pointer(h)) }
func (h *header) setI16(i int, x int16) { h.int16s()[i] = x }
func (h *header) getI16(i int) int16    { return h.int16s()[i] }

/* int32 */

func (h *header) int32s() []int32       { return *(*[]int32)(unsafe.Pointer(h)) }
func (h *header) setI32(i int, x int32) { h.int32s()[i] = x }
func (h *header) getI32(i int) int32    { return h.int32s()[i] }

/* int64 */

func (h *header) int64s() []int64       { return *(*[]int64)(unsafe.Pointer(h)) }
func (h *header) setI64(i int, x int64) { h.int64s()[i] = x }
func (h *header) getI64(i int) int64    { return h.int64s()[i] }

/* uint */

func (h *header) uints() []uint      { return *(*[]uint)(unsafe.Pointer(h)) }
func (h *header) setU(i int, x uint) { h.uints()[i] = x }
func (h *header) getU(i int) uint    { return h.uints()[i] }

/* uint8 */

func (h *header) uint8s() []uint8      { return *(*[]uint8)(unsafe.Pointer(h)) }
func (h *header) setU8(i int, x uint8) { h.uint8s()[i] = x }
func (h *header) getU8(i int) uint8    { return h.uint8s()[i] }

/* uint16 */

func (h *header) uint16s() []uint16      { return *(*[]uint16)(unsafe.Pointer(h)) }
func (h *header) setU16(i int, x uint16) { h.uint16s()[i] = x }
func (h *header) getU16(i int) uint16    { return h.uint16s()[i] }

/* uint32 */

func (h *header) uint32s() []uint32      { return *(*[]uint32)(unsafe.Pointer(h)) }
func (h *header) setU32(i int, x uint32) { h.uint32s()[i] = x }
func (h *header) getU32(i int) uint32    { return h.uint32s()[i] }

/* uint64 */

func (h *header) uint64s() []uint64      { return *(*[]uint64)(unsafe.Pointer(h)) }
func (h *header) setU64(i int, x uint64) { h.uint64s()[i] = x }
func (h *header) getU64(i int) uint64    { return h.uint64s()[i] }

/* uintptr */

func (h *header) uintptrs() []uintptr         { return *(*[]uintptr)(unsafe.Pointer(h)) }
func (h *header) setUintptr(i int, x uintptr) { h.uintptrs()[i] = x }
func (h *header) getUintptr(i int) uintptr    { return h.uintptrs()[i] }

/* float32 */

func (h *header) float32s() []float32     { return *(*[]float32)(unsafe.Pointer(h)) }
func (h *header) setF32(i int, x float32) { h.float32s()[i] = x }
func (h *header) getF32(i int) float32    { return h.float32s()[i] }

/* float64 */

func (h *header) float64s() []float64     { return *(*[]float64)(unsafe.Pointer(h)) }
func (h *header) setF64(i int, x float64) { h.float64s()[i] = x }
func (h *header) getF64(i int) float64    { return h.float64s()[i] }

/* complex64 */

func (h *header) complex64s() []complex64   { return *(*[]complex64)(unsafe.Pointer(h)) }
func (h *header) setC64(i int, x complex64) { h.complex64s()[i] = x }
func (h *header) getC64(i int) complex64    { return h.complex64s()[i] }

/* complex128 */

func (h *header) complex128s() []complex128   { return *(*[]complex128)(unsafe.Pointer(h)) }
func (h *header) setC128(i int, x complex128) { h.complex128s()[i] = x }
func (h *header) getC128(i int) complex128    { return h.complex128s()[i] }

/* string */

func (h *header) strings() []string      { return *(*[]string)(unsafe.Pointer(h)) }
func (h *header) setStr(i int, x string) { h.strings()[i] = x }
func (h *header) getStr(i int) string    { return h.strings()[i] }

/* unsafe.Pointer */

func (h *header) unsafePointers() []unsafe.Pointer         { return *(*[]unsafe.Pointer)(unsafe.Pointer(h)) }
func (h *header) setUnsafePointer(i int, x unsafe.Pointer) { h.unsafePointers()[i] = x }
func (h *header) getUnsafePointer(i int) unsafe.Pointer    { return h.unsafePointers()[i] }
