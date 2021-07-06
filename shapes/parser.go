package shapes

import (
	"bytes"
	"fmt"
	"strconv"
	"unicode"

	"github.com/pkg/errors"
)

// Parse parses a string and returns a shape expression.
func Parse(a string) (retVal Expr, err error) {
	q, err := lex(a)
	if err != nil {
		return nil, err
	}
	p := newParser(true)
	defer func() {
		if r := recover(); r != nil {
			p.printTab(nil)
			panic(r)
		}
	}()

	err = p.parse(q)
	p.logstate()
	if len(p.stack) <= 0 {
		p.printTab(nil)
		return nil, errors.Errorf("WTF?")
	}

	p.printTab(nil)
	var ok bool
	if retVal, ok = p.stack[0].(Expr); !ok {
		return nil, errors.Errorf("Expected the final parse to be an Expr. Got %v of %T instead", p.stack[0], p.stack[0])
	}
	return
}

type parser struct {
	queue      []tok           // incoming string of tokens
	stack      []substitutable // "working" stack
	infixStack []tok           // stack of operators
	qptr       int             // queue pointer

	log *bytes.Buffer
}

func newParser(log bool) *parser {
	var l *bytes.Buffer
	if log {
		l = new(bytes.Buffer)
	}
	return &parser{
		log: l,
	}
}

func (p *parser) pop() substitutable {
	if len(p.stack) == 0 {
		panic("cannot pop")
	}

	retVal := p.stack[len(p.stack)-1]
	p.stack = p.stack[:len(p.stack)-1]
	return retVal
}

func (p *parser) popExpr() (Expr, error) {
	s := p.pop()
	e, ok := s.(Expr)
	if !ok {
		return nil, errors.Errorf("Expected an Expr. Got %v of %T instead", s, s)
	}
	return e, nil
}

func (p *parser) push(a substitutable) { p.stack = append(p.stack, a) }

func (p *parser) pushInfix(t tok) { p.infixStack = append(p.infixStack, t) }

func (p *parser) popInfix() tok {
	if len(p.infixStack) == 0 {
		panic("Cannot pop infix stack")
	}
	retVal := p.infixStack[len(p.infixStack)-1]
	p.infixStack = p.infixStack[:len(p.infixStack)-1]
	return retVal
}

func (p *parser) cur() tok {
	// if p.qptr < 0 || p.qptr >= len(p.queue) {
	// 	return tok{}, errors.Errorf("Cannot get current token. Pointer: %d. Queue: %v", p.qptr, len(p.queue))
	// }
	return p.queue[p.qptr]
}

// compose performs f() then g()
func (p *parser) compose(g, f func() error, gname, fname string) func() error {
	return func() error {
		if err := f(); err != nil {
			return errors.Wrapf(err, "In composed functions %v ∘ %v. %v failed", fname, gname, fname)
		}
		if err := g(); err != nil {
			return errors.Wrapf(err, "In composed functions %v ∘ %v. %v failed", fname, gname, gname)
		}
		return nil
	}
}

// compareCur compares the cur token with  the top of the infix stack. It returns a function that the parser should take
func (p *parser) compareCur() func() error {
	p.logstate()
	t := p.cur()

	switch t.t {
	case digit:
		return p.pushNum
	case letter:
		return p.pushVar
	case axesL:
		return p.compose(p.pushVar, p.pushCurTok, "pushVar", "pushCurTok") // X is a special "variable". It's used to  mark how many items are in a axes.
	case transposeop:
		return p.resolveTranspose
	default:
		if len(p.infixStack) == 0 {
			// push the current to infixStack
			return p.pushCurTok
		}
		// check if current token has greater op prec than top of stack
		top := p.infixStack[len(p.infixStack)-1]
		topPrec := opprec[top.v]
		curPrec := opprec[t.v]

		// if current is negative, we need to resolve until the infixStack has len 0 then pushCurTok
		if curPrec < 0 {
			if err := p.pushCurTok(); err != nil {
				return func() error { return errors.Wrap(err, "curPrec < 0") }
			}
			return p.resolveAllInfix
		}

		if curPrec > topPrec {
			return p.pushCurTok
		}

		// check special case of arrows (which are right assoc)
		if top.t == arrow && t.t == arrow {
			return p.pushCurTok
		}

		// otherwise resolve first then pushcurtok
		return p.compose(p.pushCurTok, p.resolveInfixCompareCur, "pushCurTok", "resolveInfixCompareCur")
	}
}

func (p *parser) incrQPtr() error { p.qptr++; return nil }

// pushVar pushes a var on to the values stack.
func (p *parser) pushVar() error {
	t := p.cur()
	p.push(Var(t.v))

	// special cases: a[...] and X[...]
	if p.qptr == len(p.queue)-1 {
		return nil
	}

	next := p.queue[p.qptr+1] // peek
	switch {
	case t.v != 'X' && next.v == '[':
		p.push(SliceOf{})
		// consume the '[' token
		p.incrQPtr()
		p.pushInfix(p.queue[p.qptr])
	case t.v == 'X' && next.v != '[':
		// error
		return errors.Errorf("Expected '[' after X. Got %v instead", next)
	case t.v == 'X' && next.v == '[':
		// consume the '[' token
		p.incrQPtr()
		p.pushInfix(p.queue[p.qptr])
	}

	return nil
}

// pushNum pushes a number (typed as a Size) onto the values stack.
func (p *parser) pushNum() error {
	t := p.cur()
	p.push(Size(int(t.v)))
	return nil
}

// pushCurTok pushes the current token into the infixStack
func (p *parser) pushCurTok() error {
	t := p.cur()
	p.pushInfix(t)
	return nil
}

// checkItems checks that there are at least `expected` number of items in the stack
func (p *parser) checkItems(expected int) error {
	if len(p.stack) < expected {
		return errors.Errorf("Expected at least %d items in stack. Stack %v", expected, p.stack)
	}
	return nil
}

func (p *parser) parse(q []tok) (err error) {
	p.queue = q
	for p.qptr < len(p.queue) {
		if err = p.parseOne(); err != nil {
			// p.printTab(nil)
			return err
		}
		p.incrQPtr()
	}
	if len(p.infixStack) > 0 {
		if err := p.resolveAllInfix(); err != nil {
			return errors.Wrap(err, "Unable to resolve all infixes while in .parse")
		}
	}
	//p.printTab(nil)
	return nil
}

func (p *parser) parseOne() error {
	t := p.cur()

	// special cases: ()
	if t.v == '(' {
		if p.qptr == len(p.queue)-1 {
			return errors.Errorf("Dangling open paren '(' at %v", t.l)
		}
		if p.queue[p.qptr+1].v == ')' {
			p.push(Shape{})
			p.incrQPtr()
			return nil
		}
	}

	fn := p.compareCur()
	if err := fn(); err != nil {
		return errors.Wrap(err, "Unable to parseOne")
	}
	return nil
}

// resolveAllInfix resolves all the infixes in the infixStack. The result will be an empty infix stack.
func (p *parser) resolveAllInfix() error {
	var count int
	for len(p.infixStack) > 0 {
		if err := p.resolveInfix(); err != nil {
			if _, ok := err.(NoOpError); ok {
				break
			}
			return errors.Wrapf(err, "Unable to resolve all infixes. %d processed.", count)
		}
		count++
	}
	return nil
}

// resolveInfixCompareCur will resolve the infixes until such a time that the top of the infixStack has smaller precedence than the current.
func (p *parser) resolveInfixCompareCur() error {
	t := p.cur()
	top := p.infixStack[len(p.infixStack)-1]
	topPrec := opprec[top.v]
	curPrec := opprec[t.v]

	for curPrec < topPrec && curPrec >= 0 {
		p.logstate("cur %v top %v | %d %d", t, top, curPrec, topPrec)
		if err := p.resolveInfix(); err != nil {
			if _, ok := err.(NoOpError); ok {
				break // No Op error is returned when there is a paren
			}
			return errors.Wrap(err, "cannot resolveInfixCompareCur")
		}
		if len(p.infixStack) == 0 {
			break
		}
		top = p.infixStack[len(p.infixStack)-1]
		topPrec = opprec[top.v]
	}

	return nil
}

// resolveInfix resolves one infix operator from the infixStack
func (p *parser) resolveInfix() error {
	last := p.popInfix()
	p.logstate("resolveInfix %v", last)
	var err error
	switch last.t {
	case unop:
		if err = p.resolveUnOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve unop %v.", last)
		}
	case binop:
		if err = p.resolveBinOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve binop %v.", last)
		}
	case cmpop:
		if err = p.resolveCmpOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve cmpop %v.", last)
		}
	case logop:
		if err = p.resolveLogOp(last); err != nil {
			return errors.Wrapf(err, "Unable to resolve logop %v.", last)
		}
	case arrow:
		if err := p.resolveArrow(last); err != nil {
			return errors.Wrapf(err, "Cannot resolve arrow %v.", last)
		}
	case parenL:
		p.pushInfix(last)
		return noopError{}
	case parenR:
		if err := p.resolveA(); err != nil {
			return errors.Wrap(err, "Cannot resolve A.")
		}
	case comma:
		if err := p.resolveComma(last); err != nil {
			return errors.Wrapf(err, "Cannot resolve comma %v", last)
		}
	case braceR:
		if err := p.resolveCompound(); err != nil {
			return errors.Wrapf(err, "Cannot resolve compound %v", last)
		}
	case brackR:
		if err := p.resolveSlice(); err != nil {
			return errors.Wrapf(err, "Cannot resolve slice %v", last)
		}
	case colon:
		if err := p.resolveColon(); err != nil {
			return errors.Wrapf(err, "Cannot resolve colon %v", last)
		}
	case axesL:
		if err := p.resolveAxes(); err != nil {
			return errors.Wrapf(err, "Cannot resolve Axes %v", last)
		}
	default:
		//	log.Printf("last {%v %c %v} is unhandled", last.t, last.v, last.l)
	}

	return nil
}

// resolveGroup resolves groupings of `(...)` and `[...]`
func (p *parser) resolveGroup(want rune) error {
	var bw []tok
	var found bool
loop:
	for i := len(p.infixStack) - 1; i >= 0; i-- {
		t := p.popInfix()
		// keep going until you find the first '[' or '(', whatever that was passed into `want`

		switch {
		case t.v == want && want == '[':
			// special case, iterate once more to find out if the infix operator before is 'X'
			if i-1 < 0 {
				found = true
				break loop
			}

			x := p.popInfix()
			if x.v != 'X' {
				p.pushInfix(x) // undo the pop
				found = true
				break loop
			}

			bw = append(bw, x)

			fallthrough
		case t.v == want:
			found = true
			break loop
		}

		bw = append(bw, t)
	}
	if !found {
		return errors.Errorf("Could not find a corresponding %q in expression. Unable to resolveGroup. Popped Infix (in backwards order) %v", want, bw)
	}
	reverse(bw)
	backup := p.infixStack

	p.infixStack = bw
	if err := p.resolveAllInfix(); err != nil {
		return errors.Wrapf(err, "Unable to resolveGroup. Group: %q", want)
	}
	if len(p.infixStack) > 0 {
		// error? TODO
	}
	p.infixStack = backup
	return nil
}

// resolveA resolves an Abstract{}. If it can be turned into a Shape{}, then the shape will be returned.
func (p *parser) resolveA() error {
	if err := p.resolveGroup('('); err != nil {
		return errors.Wrap(err, "Unable to resolveA.")
	}

	last := p.pop()
	if abs, ok := last.(Abstract); ok {
		if shp, ok := abs.ToShape(); ok {
			p.push(shp)
			return nil
		}
	}
	// `last` is an Expr that is not a Shape or Abstract.
	p.push(last)
	return nil
}

// resolveArrow resolves an Arrow.
func (p *parser) resolveArrow(t tok) error {
	snd, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve snd of arrow at %d as Expr.", t.l)
	}
	fst, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve fst of arrow at %d as Expr.", t.l)
	}
	arr := Arrow{
		A: fst,
		B: snd,
	}
	p.push(arr)
	return nil
}

// resolveComma resolves a comma in group/shape.
func (p *parser) resolveComma(t tok) error {
	// comma is a binary option. However if it's a trailing comma, then there is no need.
	snd := p.pop()

	// If it's a trailing comma:
	// 	(a, )
	// then the stacks will look something like this:
	// 	[a] ['(' ',' ]
	// Thus after the stack is popped
	// 	[] ['(']  | WORKING: snd == 'a'
	if len(p.stack) == 0 {
		switch s := snd.(type) {
		case Abstract:
			p.push(s)
		case Sizelike:
			p.push(Abstract{s})
		default:
			return errors.Errorf("Failed to resolveComma. Cannot handle %v of %T as a Sizelike.", snd, snd)
		}
		return nil
	}

	// if the length is not 0, then there are more values to pop off the stack.
	fst := p.pop()
	switch f := fst.(type) {
	case Abstract:
		switch s := snd.(type) {
		case Sizelike:
			f = append(f, s)
			p.push(f)
			return nil
		case Conser:
			ret := f.Cons(s).(substitutable)
			p.push(ret)
			return nil
		}
	case Sizelike:
		switch s := snd.(type) {
		case Sizelike:
			p.push(Abstract{f, s})
			return nil
		case Shape:
			abs := Abstract{f}
			ret := abs.Cons(s).(substitutable)
			p.push(ret)
			return nil
		case Abstract:
			s = append(s, f)
			copy(s[1:], s[0:])
			s[0] = f
			p.push(s)
			return nil
		}
	}
	return errors.Errorf("Unable to resolveComma. Arrived at an unreachable state. Check your input expression.")
}

// resolveUnOp resolves a unary op.
func (p *parser) resolveUnOp(t tok) error {
	if err := p.checkItems(1); err != nil {
		return errors.Wrap(err, "Unable to resolveUnOp.")
	}
	expr, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Cannot resolve expr of unop %c at %d.", t.v, t.l)
	}
	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse UnOp OpType %v", t)
	}
	o := UnaryOp{
		Op: op,
		A:  expr,
	}
	p.push(o)
	return nil
}

// resolveBinOp resolves a binary op.
func (p *parser) resolveBinOp(t tok) error {
	if err := p.checkItems(2); err != nil {
		return errors.Wrap(err, "Unable to resolveBinOp")
	}

	snd, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Unable to resolve snd of BinOp at %d as Expr.", t.l)
	}
	fst, err := p.popExpr()
	if err != nil {
		return errors.Wrapf(err, "Unable to resolve fst of BinOp at %d as Expr.", t.l)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse BinOp OpType %v", t)
	}

	o := BinOp{
		Op: op,
		A:  fst,
		B:  snd,
	}
	p.push(o)
	return nil
}

// resolveCmpOp resolves a comparison op.
func (p *parser) resolveCmpOp(t tok) error {
	if err := p.checkItems(2); err != nil {
		return errors.Wrap(err, "Unable to resolveCmpOp")
	}

	snd := p.pop()
	sndOp, ok := snd.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve snd of CmpOp %c at %d as Operation. Got %v of %T instead ", t.v, t.l, snd, snd)
	}
	fst := p.pop()
	fstOp, ok := fst.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve fst of CmpOp %c at %d as Operation. Got %v of %T instead.", t.v, t.l, fst, fst)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse CmpOp OpType %v", t)
	}

	o := SubjectTo{
		OpType: op,
		A:      fstOp,
		B:      sndOp,
	}
	p.push(o)
	return nil

}

// resolveLogOp resolves a logical op.
func (p *parser) resolveLogOp(t tok) error {
	if err := p.checkItems(2); err != nil {
		return errors.Wrap(err, "Unable to resolveLogOp")
	}

	snd := p.pop()
	sndOp, ok := snd.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve snd of LogOp %c at %d as Operation. Got %T instead ", t.v, t.l, snd)
	}
	fst := p.pop()
	fstOp, ok := fst.(Operation)
	if !ok {
		return errors.Errorf("Cannot resolve fst of LogOp %c at %d as Operation. Got %v of %T instead", t.v, t.l, fst, fst)
	}

	op, err := parseOpType(t.v)
	if err != nil {
		return errors.Wrapf(err, "Unable to parse LogOp OpType %v", t)
	}

	o := SubjectTo{
		OpType: op,
		A:      fstOp,
		B:      sndOp,
	}
	p.push(o)
	return nil
}

// resolveCompound expects the stack to look like this:
// 	[..., Expr, SubjectTo{...}]
// The result will look like this
// 	[..., Compound{...}] (the Compound{} now has data)
func (p *parser) resolveCompound() error {
	if err := p.checkItems(2); err != nil {
		return errors.Wrap(err, "Unable to resolveCompound")
	}
	
	// find '{'
	var found bool
	for i := len(p.infixStack)-1; i>= 0; i--{
		if p.infixStack[i].t == braceL{
			found = true
			break
		}
	}
	if !found {
return errors.Errorf("Unable to resolveCompound. Received a '}' with no preceeding '{'")
	}

	// first check
	var st SubjectTo
	var e Expr
	var ok bool

	top := p.pop() // SubjectTo
	if st, ok = top.(SubjectTo); !ok {
		return errors.Errorf("Expected Top of Stack to be a SubjectTo is %v of %T. Stack: %v", top, top, p.stack)
	}
	snd := p.pop() // Expr
	if e, ok = snd.(Expr); !ok {
		return errors.Errorf("Expected Second of Stack to be a Expr is %v of %T. Stack: %v", snd, snd, p.stack)
	}
	c := Compound{
		Expr:      e,
		SubjectTo: st,
	}

	p.push(c)
	return nil
}

// resolveSlice resolves a slice. It calls resolveGroup().
func (p *parser) resolveSlice() error {
	// five cases:
	// 1. single slice (e.g. a[0])
	// 2. range (e.g. a[0:2])
	// 3. stepped range (e.g. a[0:2:2])
	// 4. open range (e.g. a[1:]) CURRENTLY UNSUPPORTED. TODO.
	// 5. limit range (e.g. a[:2]) CURRENTLY UNSUPPORTED. TODO.

	// pop infixStack - this will handle any of the cases with colons.
	if err := p.resolveGroup('['); err != nil {
		return errors.Wrap(err, "Unable to resolveSlice.")
	}

	// resolve any potential SliceOf{} or IndexOf{}
	top := p.pop()

	// check for case 1
	var idxof bool
	var idx int
	var slice Sli
	switch t := top.(type) {
	case Sli:
		slice = t
	case Size:
		idx = int(t)
		idxof = true
	case Axes:
		// oops it's not actually a slice
		p.push(top)
		return nil
	default:
		return errors.Errorf("top can either be Sli or Size. Got %v of %T instead", top, top)
	}

	snd := p.pop()
	so, ok := snd.(SliceOf)
	if !ok {
		p.push(snd)
		// check if idxof is true
		if idxof {
			// top should no longer just be an int
			top = Sli{start: idx, end: idx + 1, step: 1}
		}
		p.push(top)
		return nil
	}

	// if it's ok, then the third from the stack would be an Expr
	thd := p.pop().(Expr)
	if idxof {
		// then use IndexOf instead of SliceOf
		iof := IndexOf{I: Size(idx), A: thd}
		p.push(iof)
		return nil
	}
	so.Slice = slice
	so.A = thd
	p.push(so)
	return nil
}

// resolveColon resolves a colon much like resolveComma.
func (p *parser) resolveColon() error {
	// five cases:
	// 1. single slice (e.g. a[0])
	// 2. range (e.g. a[0:2])
	// 3. stepped range (e.g. a[0:2:2])
	// 4. open range (e.g. a[1:]) CURRENTLY UNSUPPORTED. TODO.
	// 5. limit range (e.g. a[:2]) CURRENTLY UNSUPPORTED. TODO.

	if err := p.checkItems(2); err != nil {
		return errors.Wrap(err, "Unable to resolveColon.")
	}

	// a colon is a binop
	snd := p.pop()
	fst := p.pop()

	switch s := snd.(type) {
	case Size:
		// case 2
		f, ok := substToInt(fst)
		if !ok {
			return errors.Errorf("Expected fst to be a Size. Got %v of %T instead", fst, fst)
		}
		retVal := Sli{start: f, end: int(s), step: 1}
		p.push(retVal)
	case Sli:
		// case 3
		f, ok := substToInt(fst)
		if !ok {
			return errors.Errorf("Expected fst to be a Size. Got %v of %T instead", fst, fst)
		}
		s.step = s.end
		s.end = s.start
		s.start = f
		p.push(s)
	default:
		return errors.Errorf("Unsupported: case 4 and 5")

	}
	return nil
}
func (p *parser) resolveAxes() error {
	var bw Axes
	for i := len(p.stack) - 1; i >= 0; i-- {
		t := p.pop()
		if v, ok := t.(Var); ok && v == Var('X') {
			break
		}
		ax, ok := substToInt(t)
		if !ok {
			return errors.Errorf("Failed to resolveAxes. %dth item in stack is expected to be an int-like. Got %v of %T instead", i, t, t)
		}
		bw = append(bw, Axis(ax))
	}
	reverseAxes(bw)
	p.push(bw)
	return nil
}

// resolveTranspose is a janky way of resolving a transpose operator.
func (p *parser) resolveTranspose() error {
	p.incrQPtr()
	backup1 := p.stack
	backup2 := p.infixStack

	// check
	if len(p.queue) <= p.qptr {
		return errors.Errorf("Dangling T operator")
	}

	p.stack = nil
	p.infixStack = nil
	if err := p.expectAxes(); err != nil {
		return errors.Wrapf(err, " failed to transpose")
	}
	axes := p.stack[len(p.stack)-1].(Axes)

	p.stack = nil
	p.infixStack = nil
	if err := p.expectExpr(); err != nil {
		return errors.Wrap(err, "failed to transposeOf")
	}

	A := p.stack[len(p.stack)-1].(Expr)

	p.stack = backup1
	p.infixStack = backup2
	p.push(TransposeOf{Axes: axes, A: A})
	return nil

}

// expectAxes expects the next expression in the queue to be an Axes. Used only for transpose
func (p *parser) expectAxes() error {
	x := p.cur()
	if x.v != 'X' {
		return errors.Errorf("Expected 'X'. Got %q instead", x.v)
	}
	p.incrQPtr()

	if len(p.queue) <= p.qptr {
		return errors.Errorf("Dangling X.")
	}

	lbrack := p.cur()
	if lbrack.v != '[' {
		return errors.Errorf("Expected '[. Got %q instead.", lbrack.v)
	}
	p.incrQPtr()

	var axes Axes
	for next := p.cur(); next.v != ']'; next = p.cur() {
		if next.t != digit {
			// TODO: error?
		}
		axes = append(axes, Axis(next.v))
		p.incrQPtr()
	}
	p.incrQPtr() // because this was not automatically incremented
	p.push(axes)
	return nil
}

// expectExpr parses the next Expr from the queue. Used only for transpose.
func (p *parser) expectExpr() error {
	for {
		if err := p.parseOne(); err != nil {
			return errors.Wrap(err, "Failed to expect Expr")
		}

		if len(p.infixStack) == 0 {
			if _, ok := p.stack[0].(Expr); ok {
				break
			}
		}
		p.incrQPtr() // because this will not be automatically incremented as we are working outside the regular scheme.
	}
	return nil
}

// operator precedence table
var opprec = map[rune]int{
	'(': 70,
	')': -1,
	'[': 1,
	']': 70,
	'{': 70,
	'}': -1,
	',': 2,
	':': 75,
	'|': -1,
	'→': 0,

	// unop
	'K': 60,
	'D': 60,
	'Π': 60,
	'Σ': 60,
	'∀': 60,

	// binop
	'+': 40,
	'-': 40,
	'×': 50,
	'÷': 50,

	// cmpop
	'=': 30,
	'≠': 30,
	'<': 30,
	'>': 30,
	'≤': 30,
	'≥': 30,

	// logop
	'∧': 20,
	'∨': 10,

	// axes
	'X': 60,

	// TransposeOf
	'T': 60,
}

type tokentype int

const (
	eos tokentype = iota
	parenL
	parenR
	brackL
	brackR
	axesL // use brackR for closing
	braceL
	braceR
	digit
	letter
	comma
	arrow
	colon
	pipe
	unop
	binop
	cmpop
	logop
	transposeop
)

type tok struct {
	t tokentype // type
	v rune      // value of the token
	l int       // location
}

func (t tok) Format(s fmt.State, c rune) {
	switch t.t {
	case letter:
		fmt.Fprintf(s, "{%c %d}", t.v, t.l)
	case digit:
		fmt.Fprintf(s, "{%d %d}", t.v, t.l)
	case eos:
		fmt.Fprintf(s, "{EOS %d}", t.l)
	default:
		fmt.Fprintf(s, "{%c %d}", t.v, t.l)
	}
}

const eoserr = "Lex Error: sudden end of string found in %q. Position %d"

// lex is a function that takes a string and returns a slice of tokens.
//
// it's fundamentally a giant state table in a for loop. Since the grammar of the language is very strict, it is a fairly straight forwards parse.
func lex(a string) (retVal []tok, err error) {
	rs := []rune(a)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case r == '(':
			retVal = append(retVal, tok{parenL, r, i})
		case r == ')':
			retVal = append(retVal, tok{parenR, r, i})
		case r == '[':
			retVal = append(retVal, tok{brackL, r, i})
		case r == ']':
			retVal = append(retVal, tok{brackR, r, i})
		case r == '{':
			retVal = append(retVal, tok{braceL, r, i})
		case r == '}':
			retVal = append(retVal, tok{braceR, r, i})
		case r == '→':
			retVal = append(retVal, tok{arrow, r, i})
		case r == '-':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '>' {
				i++
				retVal = append(retVal, tok{arrow, '→', i})
				continue
			}
			retVal = append(retVal, tok{binop, r, i})
		case r == '+', r == '*', r == '×', r == '/', r == '∕', r == '÷':
			rr := r
			if r == '*' {
				rr = '×'
			}
			if r == '/' || r == '∕' {
				rr = '÷'
			}
			retVal = append(retVal, tok{binop, rr, i})
		case r == '=', r == '≠', r == '≥', r == '≤', r == '≥', r == '≤': // single symbol cmp op (note there are TWO acceptable unicode symbols for gte and lte)
			retVal = append(retVal, tok{cmpop, r, i})
		case r == '!':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '=' {
				i++
				retVal = append(retVal, tok{cmpop, '≠', i})
			}
		case r == '>', r == '<':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '=' {
				i++
				var rr rune
				switch r {
				case '>':
					rr = '≥'
				case '<':
					rr = '≤'
				}
				retVal = append(retVal, tok{cmpop, rr, i})
				continue
			}
			retVal = append(retVal, tok{cmpop, r, i})
		case r == '∧', r == '∨', r == '⋀', r == '⋁': // single symbol logical op
			retVal = append(retVal, tok{logop, r, i})
		case r == '&':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '&' { // for people who think AND is written "&&"
				i++
				retVal = append(retVal, tok{logop, '∧', i})
				continue
			}
		case r == '|':
			if i+1 >= len(rs) {
				return nil, errors.Errorf(eoserr, a, i)
			}
			if rs[i+1] == '|' { // for people who think OR is written "||"
				i++
				retVal = append(retVal, tok{logop, '∨', i})
				continue
			}
			retVal = append(retVal, tok{pipe, r, i})
		case r == 'Π', r == 'Σ', r == '∀', r == 'D', r == 'P', r == 'S':
			rr := r
			if r == 'P' {
				rr = 'Π'
			}
			if r == 'S' {
				rr = 'Σ'
			}
			retVal = append(retVal, tok{unop, rr, i})
		case r == 'T':
			retVal = append(retVal, tok{transposeop, r, i})
		case r == 'X':
			retVal = append(retVal, tok{axesL, r, i})
		case r == ',':
			retVal = append(retVal, tok{comma, r, i})
		case r == ':':
			retVal = append(retVal, tok{colon, r, i})
		case unicode.IsSpace(r):
			continue // we ignore spaces and delimiters
		case unicode.IsDigit(r):
			var rs2 = []rune{r}
			var j int
			for j = i + 1; j < len(rs); j++ {
				r2 := rs[j]
				if !unicode.IsDigit(r2) {
					i = j - 1
					break
				}
				rs2 = append(rs2, r2)
			}
			i = j - 1
			s := string(rs2)
			num, err := strconv.Atoi(s)
			if err != nil {
				return nil, errors.Wrapf(err, "Unable to parse %q as a number", s)
			}

			retVal = append(retVal, tok{digit, rune(int32(num)), i})
		case unicode.IsLetter(r):
			retVal = append(retVal, tok{letter, r, i})
			// check if next is a letter. if it is  then error out
			if i+1 >= len(rs) {
				return retVal, nil
			}
			if unicode.IsLetter(rs[i+1]) {
				return nil, errors.Errorf("Only single letters are allowed as variables.")
			}

		}

	}
	return retVal, nil
}
