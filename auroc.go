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
}

// AUCroc is returned by AreaUnderROC() call.
type AUCroc struct {
	TargetRange TargetRange
	Auc1        float64
	Auc0        float64
	U1          float64
	U0          float64
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
