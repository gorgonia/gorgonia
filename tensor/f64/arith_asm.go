// +build sse avx

package tensorf64

func vecAdd(a, b []float64)
func vecSub(a, b []float64)
func vecMul(a, b []float64)
func vecDiv(a, b []float64)

func vecSqrt(a []float64)
func vecInvSqrt(a []float64)

/*
func vecPow(a, b []float64)
*/

/*
func vecScale(s float64, a []float64)
func vecScaleFrom(s float64, a []float64)
func vecTrans(s float64, a []float64)
func vecTransFrom(s float64, a []float64)
func vecPower(s float64, a []float64)
func vecPowerFrom(s float64, a []float64)
*/
