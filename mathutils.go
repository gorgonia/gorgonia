// +build !amd64

package gorgonia

import "math/big"

func divmod(a, b int) (int, int) {
	q := new(big.Int)
	r := new(big.Int)

	aI := big.NewInt(int64(a))
	bI := big.NewInt(int64(b))
	q.DivMod(aI, bI, r)
	return int(q.Int64()), int(r.Int64())
}
