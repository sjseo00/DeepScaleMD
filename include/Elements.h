#pragma once

#include <string>

// Atomic number of an element symbol written the usual way: one capital letter followed by lower-case ones ("H", "O", "Na",
// "Cl"). Returns 0 when the text is not an element symbol (so "OW", "CA" and "na" are not: only the exact spelling counts).
int atomic_number_from_symbol(const std::string& symbol);

// Symbol of the element with atomic number z (1..118), "" for any other number.
std::string element_symbol(int z);

// The element whose standard atomic mass (hydrogen 1.008, carbon 12.011, oxygen 15.999, ...; elements 1..94) is within `tolerance` u of
// `mass`: the atomic number, or 0 when no element is that close or two are (a topology with rounded masses such as 16.0 still works;
// a hydrogen with a repartitioned mass of 3.024 does not match anything).
int atomic_number_from_mass(double mass, double tolerance = 0.1);
