// +build !debug

package tensor

const DEBUG = false

var TABCOUNT uint32 = 0

func tabcount() int { return 0 }

func enterLoggingContext()                      {}
func leaveLoggingContext()                      {}
func logf(format string, others ...interface{}) {}
func loggc()                                    {}
