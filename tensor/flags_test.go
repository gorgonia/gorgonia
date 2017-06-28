package tensor

import "testing"

func TestMemoryFlag(t *testing.T) {
	var defaultFlag MemoryFlag
	if defaultFlag.manuallyManaged() || !defaultFlag.nativelyAccessible() {
		t.Errorf("Something went wrong with the creation of flags")
	}

	a := ManuallyManaged
	if !a.manuallyManaged() {
		t.Errorf("Expected ManuallyManaged to be true")
	}
	if !a.nativelyAccessible() {
		t.Errorf("Expected ManuallyManaged to be nativelyAccessible")
	}

	b := NativelyInaccessible
	if b.manuallyManaged() {
		t.Errorf("Expected NativelyInaccessible to not be manually managed")
	}
	if b.nativelyAccessible() {
		t.Errorf("Expected NativelyInaccessible to be false %v", b.nativelyAccessible())
	}

	c := MakeMemoryFlag(ManuallyManaged, NativelyInaccessible)
	if !c.manuallyManaged() {
		t.Errorf("Expected c to be manually managed")
	}
	if c.nativelyAccessible() {
		t.Errorf("Expected c to be natively inaccessible")
	}
}

func TestDataOrder(t *testing.T) {
	var defaultFlag DataOrder
	if defaultFlag.isColMajor() || defaultFlag.isNotContiguous() {
		t.Errorf("Expected default flag to be row major and contiguous")
	}
	if !(defaultFlag.isRowMajor() && defaultFlag.isContiguous()) {
		t.Errorf("Expected default flag to be row major and contiguous")
	}

	cm := ColMajor
	if cm.isRowMajor() {
		t.Errorf("colMajor cannot be rowMajor")
	}
	if cm.isNotContiguous() {
		t.Errorf("ColMajor by default is contiguous")
	}

	// check toggle
	rm := cm.toggleColMajor()
	if rm.isColMajor() {
		t.Errorf("toggled cm should be rm")
	}

	cm = rm.toggleColMajor()
	if cm.isRowMajor() {
		t.Errorf("toggled rm should be cm")
	}
}
