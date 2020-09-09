package dataviz

import (
	"encoding/json"
	"io"
	"strconv"
	"time"

	"gorgonia.org/gorgonia"
	xvm "gorgonia.org/gorgonia/x/vm"
)

// DumpTrace suitable for https://github.com/vasturiano/timelines-chart
func DumpTrace(traces []xvm.Trace, g *gorgonia.ExprGraph, w io.Writer) error {
	var zerotime time.Time
	groups := make(map[string]group)
	// generate all labels
	for _, trace := range traces {
		if trace.End == zerotime {
			continue
		}
		if _, ok := groups[trace.StateFunction]; !ok {
			groups[trace.StateFunction] = group{
				Group: trace.StateFunction,
			}
		}
		label := dataLabel{
			TimeRange: []time.Time{
				trace.Start,
				trace.End,
			},
			Val: strconv.Itoa(int(trace.ID)),
		}
		dGroup := dataGroup{
			Label: g.Node(trace.ID).(*gorgonia.Node).Name(),
			Data: []dataLabel{
				label,
			},
		}
		g := groups[trace.StateFunction]
		g.Data = append(g.Data, dGroup)
		groups[trace.StateFunction] = g
	}
	grps := make([]group, 0, len(groups))
	for _, grp := range groups {
		grps = append(grps, grp)
	}
	enc := json.NewEncoder(w)
	return enc.Encode(grps)
}

type group struct {
	Group string      `json:"group"`
	Data  []dataGroup `json:"data"`
}

type dataGroup struct {
	Label string      `json:"label"`
	Data  []dataLabel `json:"data"`
}

type dataLabel struct {
	TimeRange []time.Time `json:"timeRange"`
	Val       interface{} `json:"val"`
}
