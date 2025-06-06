#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#
include make.def

EXES = activate_function$(EXE)

all: $(EXES)

activate_function$(EXE): activate_function.$(OBJ)
	nvc++ $(OPTFLAGS) -o activate_function$(EXE) activate_function.$(OBJ)

test: $(EXES)
	for i in $(EXES); do \
            $(PRE)$$i; \
        done

clean:
	$(RM) $(EXES) *.$(OBJ) *.ptx *.cub

mm_utils.$(OBJ): mm_utils.h

.SUFFIXES:
.SUFFIXES: .c .cpp  .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	#$(CC) $(CFLAGS) -c $<
	nvc++ $(CFLAGS) -c $<