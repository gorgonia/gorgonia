package gorgonia

import (
	"fmt"
	"math"
	"sort"
)

// TargetRange defines how to map a real target feature
// to a category/class.
type TargetRange struct {
	// We just have just one threshold for now, but could be a range
	// or one range vs. another for ordinal/multiclass disrimination.
	Thresh float64

	// should we compute the U_R objective function, a differentiable
	// approximation to the Area under ROC, but designed to be minimized.
	//
	// Not implemented: actual optimization by using the derivative
	// of the objective with respect to inputs X. Hence think of
	// this simply as a proof of concept to allow comparison of
	// the actual ROC and the approximation function to check for
	// applicability on a given data context.
	//
	ComputeObjective bool
	// If ComputeObjective is true, then Margin and Power must be set.
	Margin float64 // Margin must be > 0, and Margin must <= 1.  Starting guess might be 0.2
	Power  float64 // Power must be > 1. Starting guess might be 3.0
}

// AUCroc is returned by AreaUnderROC() call.
type AUCroc struct {
	TargetRange TargetRange
	Auc1        float64
	Auc0        float64
	U1          float64
	U0          float64

	N1 float64
	N0 float64

	// Objective function value, set if TargetRrange.ComputeObjective is true.
	Obj float64
}

// AreaUnderROC() is for
// classifying a target real values as <= targetRanges[i].Threshold
// versus target > targetRanges[i].Threshold,
// for each i over the supplied targetRanges.
// The AUC for ROC is equivalent to the Wilcoxon or Mann-Whitney U test statistic with the relation:
//
// AUC = U/(n0 * n1)
//
// No NaN handling at present so handle those prior. This could be added
// without too much difficulty. VposSlice will sort them to the end, but
// another pass would be needed to keep pair-wise complete target/predictor pairs.
//
// returns:
// auc1 is the area under the curve for classifying target > targetRanges[i].Threshold (typically what one wants).
// auc0 is the area under the curve for classifying target <= targetRanges[i].Threshold.
func AreaUnderROC(predictor, target []float64, targetRanges []*TargetRange) (res []*AUCroc, err error) {
	n := len(target)
	npred := len(predictor)
	if npred != n {
		panic(fmt.Sprintf("predictor len %v not equal target len %v", npred, n))
	}

	// order the predictors
	vp := make([]vpos, n)
	for i, x := range predictor {
		vp[i].val = x
		vp[i].pos = i
		if math.IsNaN(x) {
			panic(fmt.Sprintf("nan not handled in AreaUnderROC(). NaN seen at pos %v in predictor", i))
		}
	}
	vs := VposSlice(vp)
	sort.Sort(vs)

	// get the rank of each prediction  (will be .pos +1); and
	// these are back in correspondence to the target.
	rank := make([]vpos, n)
	for i, x := range vp {
		rank[i].val = float64(x.pos)
		rank[i].pos = i
	}
	rs := VposSlice(rank)
	sort.Sort(rs)

	if len(targetRanges) == 0 {
		targetRanges = []*TargetRange{&TargetRange{Thresh: 0}}
	}
	res = make([]*AUCroc, len(targetRanges))
	for j := range res {
		roc := &AUCroc{
			TargetRange: *targetRanges[j],
		}
		res[j] = roc

		var isClass1 []bool
		computeObj := roc.TargetRange.ComputeObjective
		if computeObj {
			isClass1 = make([]bool, len(target))
		}

		sum1 := 0
		sum0 := 0
		n1int := 0
		th := roc.TargetRange.Thresh

		for i, y := range target {
			if math.IsNaN(y) {
				panic(fmt.Sprintf("nan not handled in AreaUnderROC(). NaN seen at pos %v in target.", i))
			}
			if y > th {
				// U statistic assumes ranks start at 1, so add 1 here.
				sum1 += rs[i].pos + 1
				n1int++
				if computeObj {
					isClass1[i] = true
				}

			} else {
				sum0 += rs[i].pos + 1
			}
		}
		n1 := float64(n1int)
		n0 := float64(n - n1int)

		roc.U1 = float64(sum1) - n1*(n1+1)/2
		roc.U0 = float64(sum0) - n0*(n0+1)/2

		roc.Auc1 = roc.U1 / (n1 * n0)
		roc.Auc0 = roc.U0 / (n1 * n0)

		roc.N1 = n1
		roc.N0 = n0

		obj := 0.0
		if computeObj {
			power := roc.TargetRange.Power
			margin := roc.TargetRange.Margin
			if power <= 1.0 {
				panic("power must be > 1.0")
			}
			if margin <= 0 || margin > 1 {
				panic("must have 0 < margin <= 1")
			}

			for i, iIsClass1 := range isClass1 {
				for j, jIsClass1 := range isClass1 {
					if i == j {
						continue // comparison to self irrelevant
					}
					if jIsClass1 {
						continue // j only sums over negative examples or class 0.
					}
					if !iIsClass1 {
						continue // i only sums over positive examples or class 1.
					}
					pred1 := predictor[i] // one prediction when target is class 1.
					pred0 := predictor[j] // one prediction when target is class 0.

					// This is the U_R objective seen in Equation 8 of Yan et al 2003.
					obj += R1(pred1, pred0, power, margin)
				}
			}
		}
		roc.Obj = obj
	}
	return
}

// vpos holds a value and its original position
// in the array, prior to sorting.
type vpos struct {
	val float64
	pos int
}

func (v *vpos) String() (r string) {
	return fmt.Sprintf("&vpos{val:%v, pos:%v}", v.val, v.pos)
}

// VposSlice facilitates sorting.
type VposSlice []vpos

func (p VposSlice) Len() int { return len(p) }
func (p VposSlice) Less(i, j int) bool {
	if math.IsNaN(p[i].val) {
		// sort NaN to the back
		return false
	}
	if math.IsNaN(p[j].val) {
		// sort NaN to the back
		return true
	}
	return p[i].val < p[j].val
}
func (p VposSlice) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p VposSlice) String() (r string) {
	r = "VposSlice{\n"
	for i := range p {
		r += fmt.Sprintf("%v,\n", p[i])
	}
	r += "}"
	return
}

// R1 is the minimization inner objective function
// (equation 7) from
// "Optimizing Classifier Performance via an Approximation
//  to the Wilcoxon-Mann-Whitney Statistic"
// by Yan, Dodier, Mozer and Wolniewicz,
// Proceedings of ICML 2003.
//
// It is designed to be differentiable without too much difficulty.
//
// pred1 = a prediction for a known target of 1
// pred0 = a prediction for a known target of 0
//
// The two meta-parameters for the optimization are margin and power:
// a) 0 < margin <= 1 improves generalization; and
// b) power > 1 controls how strongly the margin minus the difference
// between pred1 and pred0 is amplified;
// the margin-difference is raised to power before
// being returned.
//
// In examples they try margin = 0.3 and power = 2; or
// margin = 0.2 and power = 3; but
// these should be optimized in an outer loop.
//
// Yan: "In general, we can choose a value between 0.1
// and 0.7 for margin. Also, we have found that power = 2 or 3
// achieves similar, and generally the best results."
//
// INVAR: power > 1.0
// INVAR: 0 < margin <= 1
// These invariants should be checked by the caller for speed and
// to allow potential inlning. They are not checked within R1().
//
func R1(pred1, pred0, power, margin float64) float64 {
	diff := pred1 - pred0
	if diff >= margin {
		return 0
	}
	tmp := margin - diff // notice it will always be positive since diff < margin now.
	switch power {
	case 2.0:
		return tmp * tmp
	case 3.0:
		return tmp * tmp * tmp
	case 4.0:
		return tmp * tmp * tmp * tmp
	}
	return math.Pow(tmp, power)
}
