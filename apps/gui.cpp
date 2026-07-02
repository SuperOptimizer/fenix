// gui.cpp — THE Qt translation unit (the GUI firewall, mirroring ml/inference.cpp for libtorch):
// compiled into the fenix binary only under -DFENIX_GUI, alongside apps/driver.cpp. Qt/VTK headers are
// parsed HERE and nowhere else — the driver TU (unity or split) stays Qt-free, and the heavy Qt parse
// compiles in parallel with the rest. The static registrar below adds the `view` stage; the object is
// linked directly into the executable, so it is never dropped.
#include "gui/gui.hpp"
