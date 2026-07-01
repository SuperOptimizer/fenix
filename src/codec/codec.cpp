// src/codec/codec.cpp — the "codec" module compiled as its OWN translation unit for the SPLIT dev build
// (CMake FENIX_SPLIT=ON). Emits this module inline definitions and runs its stage self-registration in
// this TU; the driver links all the module objects together. In the default UNITY build these files are
// NOT compiled — apps/driver.cpp includes the umbrella header and is the single translation unit.
//
// This TU also provides the ONE explicit instantiation of the DCT/rANS tile codec (the split-build codec
// firewall in dct_block.hpp): FENIX_CODEC_INSTANTIATE flips its extern-template decls to definitions here,
// so every other split TU links against this copy instead of re-optimizing the ~350-line bodies.
#define FENIX_CODEC_INSTANTIATE
#include "codec/codec.hpp"
