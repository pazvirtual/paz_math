PROJNAME := PAZ_Math
LIBNAME := $(shell echo $(PROJNAME) | sed 's/_//g' | tr '[:upper:]' '[:lower:]')
ifeq ($(OS), Windows_NT)
    LIBPATH := /mingw64/lib
    INCLPATH := /mingw64/include
    OSPRETTY := Windows
else
    ifeq ($(shell uname -s), Darwin)
        OSPRETTY := macOS
    else
        OSPRETTY := Linux
    endif
    LIBPATH := /usr/local/lib
    INCLPATH := /usr/local/include
endif
CXXVER := 17
OPTIM := fast
ZIPNAME := $(PROJNAME)-$(OSPRETTY)
ifeq ($(OSPRETTY), Windows)
    ZIPCONTENTS := $(PROJNAME) lib$(LIBNAME).a
else
    ZIPCONTENTS := $(PROJNAME) lib$(LIBNAME).a
endif
CFLAGS := -O$(OPTIM) -Wall -Wextra -Wno-missing-braces
ifeq ($(OSPRETTY), macOS)
    CFLAGS += -mmacosx-version-min=10.11 -Wunguarded-availability -Wno-unknown-warning-option
else
    ifeq ($(OSPRETTY), Windows)
        CFLAGS += -Wno-cast-function-type
    endif
endif
#CXXFLAGS := -std=c++$(CXXVER) $(CFLAGS) -Wold-style-cast -IEigen
CXXFLAGS := -std=c++$(CXXVER) $(CFLAGS) -IEigen
ifeq ($(OSPRETTY), Windows)
    CXXFLAGS += -Wno-deprecated-copy
endif
ARFLAGS := -rcs

SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)

REINSTALLHEADER := $(shell cmp -s $(PROJNAME) $(INCLPATH)/$(PROJNAME); echo $$?)

print-% : ; @echo $* = $($*)

.PHONY: test
default: test

lib$(LIBNAME).a: $(OBJ)
	$(RM) lib$(LIBNAME).a
	ar $(ARFLAGS) lib$(LIBNAME).a $^

ifneq ($(REINSTALLHEADER), 0)
install: $(PROJNAME) lib$(LIBNAME).a
	cp $(PROJNAME) $(INCLPATH)/
	cp lib$(LIBNAME).a $(LIBPATH)/
else
install: $(PROJNAME) lib$(LIBNAME).a
	cp lib$(LIBNAME).a $(LIBPATH)/
endif

test: lib$(LIBNAME).a
	$(MAKE) -C test
	test/test

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	$(RM) $(OBJ) lib$(LIBNAME).a
	$(MAKE) -C test clean

zip: $(ZIPCONTENTS)
	zip -j $(ZIPNAME).zip $(ZIPCONTENTS)
