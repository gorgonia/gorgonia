// +build debug
// +build !linux

package cuda

// logtid is noop in non-linux builds
func logtid(category string, logcaller int) {}
