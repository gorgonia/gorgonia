// +build noasm

package gorgonia

func divmod(a, b int) (q, r int) {
	q = a / b
	r = a % b
	return
}
