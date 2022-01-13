package main

func in(ss []string, s string) bool{
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}
