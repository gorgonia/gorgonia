// +build noasm

package types

func Divmod(a, b int) (q, r int) {
	q = a / b
	r = a % b
	return
}
