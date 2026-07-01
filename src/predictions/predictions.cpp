// src/predictions/predictions.cpp — the "predictions" module compiled as its OWN translation unit for the SPLIT dev build
// (CMake FENIX_SPLIT=ON). Emits this module inline definitions and runs its stage self-registration in
// this TU; the driver links all the module objects together. In the default UNITY build these files are
// NOT compiled — apps/driver.cpp includes the umbrella header and is the single translation unit.
#include "predictions/predictions.hpp"
