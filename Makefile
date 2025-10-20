
FC ?= gfortran
FFLAGS ?= -O2 -Wall -Wextra -std=f2008
TARGET = eval

all: $(TARGET)

$(TARGET): precip_unet_eval.f90
	$(FC) $(FFLAGS) $< -o $@

clean:
	rm -f $(TARGET) *.o *.mod
