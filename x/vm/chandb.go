package xvm

import "gorgonia.org/gorgonia"

type chanDB struct {
	// map[tail][head]
	dico map[int64]map[int64]chan gorgonia.Value
	// map[head][tail]
	reverseDico  map[int64]map[int64]chan gorgonia.Value
	inputNodeID  int64
	outputNodeID int64
}

func (c *chanDB) closeAll() {
	for i := range c.dico {
		for j := range c.dico[i] {
			close(c.dico[i][j])
		}
	}
}

// upsert the channel to the DB, if id already exists it is overwritten
func (c *chanDB) upsert(channel chan gorgonia.Value, tail, head int64) {
	if _, ok := c.dico[tail]; !ok {
		c.dico[tail] = make(map[int64]chan gorgonia.Value, 0)
	}
	if _, ok := c.reverseDico[head]; !ok {
		c.reverseDico[head] = make(map[int64]chan gorgonia.Value, 0)
	}
	c.dico[tail][head] = channel
	c.reverseDico[head][tail] = channel
}

func newChanDB() *chanDB {
	return &chanDB{
		dico:         make(map[int64]map[int64]chan gorgonia.Value, 0),
		reverseDico:  make(map[int64]map[int64]chan gorgonia.Value, 0),
		inputNodeID:  -1,
		outputNodeID: -2,
	}
}

func (c *chanDB) getAllFromTail(tail int64) []<-chan gorgonia.Value {
	edges, ok := c.dico[tail]
	if !ok {
		return nil
	}
	output := make([]<-chan gorgonia.Value, 0, len(edges))
	for _, edge := range edges {
		output = append(output, edge)
	}
	return output
}

func (c *chanDB) getAllFromHead(head int64) []chan<- gorgonia.Value {
	edges, ok := c.reverseDico[head]
	if !ok {
		return nil
	}
	output := make([]chan<- gorgonia.Value, 0, len(edges))
	for _, edge := range edges {
		output = append(output, edge)
	}
	return output
}

func (c *chanDB) getChan(tail, head int64) (chan gorgonia.Value, bool) {
	v, ok := c.dico[tail][head]
	return v, ok
}

func (c *chanDB) len() int {
	return len(c.dico)
}
