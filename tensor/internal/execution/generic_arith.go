package execution

import (
	"math"
	"math/cmplx"

	"github.com/chewxy/math32"
)

/*
GENERATED FILE (ONCE). DO NOT EDIT
*/

func AddI(a int, b int) int                         { return a + b }
func AddI8(a int8, b int8) int8                     { return a + b }
func AddI16(a int16, b int16) int16                 { return a + b }
func AddI32(a int32, b int32) int32                 { return a + b }
func AddI64(a int64, b int64) int64                 { return a + b }
func AddU(a uint, b uint) uint                      { return a + b }
func AddU8(a uint8, b uint8) uint8                  { return a + b }
func AddU16(a uint16, b uint16) uint16              { return a + b }
func AddU32(a uint32, b uint32) uint32              { return a + b }
func AddU64(a uint64, b uint64) uint64              { return a + b }
func AddF32(a float32, b float32) float32           { return a + b }
func AddF64(a float64, b float64) float64           { return a + b }
func AddC64(a complex64, b complex64) complex64     { return a + b }
func AddC128(a complex128, b complex128) complex128 { return a + b }
func AddStr(a string, b string) string              { return a + b }
func SubI(a int, b int) int                         { return a - b }
func SubI8(a int8, b int8) int8                     { return a - b }
func SubI16(a int16, b int16) int16                 { return a - b }
func SubI32(a int32, b int32) int32                 { return a - b }
func SubI64(a int64, b int64) int64                 { return a - b }
func SubU(a uint, b uint) uint                      { return a - b }
func SubU8(a uint8, b uint8) uint8                  { return a - b }
func SubU16(a uint16, b uint16) uint16              { return a - b }
func SubU32(a uint32, b uint32) uint32              { return a - b }
func SubU64(a uint64, b uint64) uint64              { return a - b }
func SubF32(a float32, b float32) float32           { return a - b }
func SubF64(a float64, b float64) float64           { return a - b }
func SubC64(a complex64, b complex64) complex64     { return a - b }
func SubC128(a complex128, b complex128) complex128 { return a - b }
func MulI(a int, b int) int                         { return a * b }
func MulI8(a int8, b int8) int8                     { return a * b }
func MulI16(a int16, b int16) int16                 { return a * b }
func MulI32(a int32, b int32) int32                 { return a * b }
func MulI64(a int64, b int64) int64                 { return a * b }
func MulU(a uint, b uint) uint                      { return a * b }
func MulU8(a uint8, b uint8) uint8                  { return a * b }
func MulU16(a uint16, b uint16) uint16              { return a * b }
func MulU32(a uint32, b uint32) uint32              { return a * b }
func MulU64(a uint64, b uint64) uint64              { return a * b }
func MulF32(a float32, b float32) float32           { return a * b }
func MulF64(a float64, b float64) float64           { return a * b }
func MulC64(a complex64, b complex64) complex64     { return a * b }
func MulC128(a complex128, b complex128) complex128 { return a * b }
func DivI(a int, b int) int                         { return a / b }
func DivI8(a int8, b int8) int8                     { return a / b }
func DivI16(a int16, b int16) int16                 { return a / b }
func DivI32(a int32, b int32) int32                 { return a / b }
func DivI64(a int64, b int64) int64                 { return a / b }
func DivU(a uint, b uint) uint                      { return a / b }
func DivU8(a uint8, b uint8) uint8                  { return a / b }
func DivU16(a uint16, b uint16) uint16              { return a / b }
func DivU32(a uint32, b uint32) uint32              { return a / b }
func DivU64(a uint64, b uint64) uint64              { return a / b }
func DivF32(a float32, b float32) float32           { return a / b }
func DivF64(a float64, b float64) float64           { return a / b }
func DivC64(a complex64, b complex64) complex64     { return a / b }
func DivC128(a complex128, b complex128) complex128 { return a / b }
func PowF32(a float32, b float32) float32           { return math32.Pow(a, b) }
func PowF64(a float64, b float64) float64           { return math.Pow(a, b) }
func PowC64(a complex64, b complex64) complex64 {
	return complex64(cmplx.Pow(complex128(a), complex128(b)))
}
func PowC128(a complex128, b complex128) complex128 { return cmplx.Pow(a, b) }
func ModI(a int, b int) int                         { return a % b }
func ModI8(a int8, b int8) int8                     { return a % b }
func ModI16(a int16, b int16) int16                 { return a % b }
func ModI32(a int32, b int32) int32                 { return a % b }
func ModI64(a int64, b int64) int64                 { return a % b }
func ModU(a uint, b uint) uint                      { return a % b }
func ModU8(a uint8, b uint8) uint8                  { return a % b }
func ModU16(a uint16, b uint16) uint16              { return a % b }
func ModU32(a uint32, b uint32) uint32              { return a % b }
func ModU64(a uint64, b uint64) uint64              { return a % b }
func ModF32(a float32, b float32) float32           { return math32.Mod(a, b) }
func ModF64(a float64, b float64) float64           { return math.Mod(a, b) }
