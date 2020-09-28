package shapes

import (
	"sort"

	"github.com/xtgo/set"
)

type varset []Var

func (vs varset) Len() int { return len(vs) }

func (vs varset) Less(i, j int) bool { return vs[i] < vs[j] }

func (vs varset) Swap(i, j int) { vs[i], vs[j] = vs[j], vs[i] }

func unique(a varset) varset {
	sort.Sort(a)
	n := set.Uniq(a)
	return a[:n]
}
