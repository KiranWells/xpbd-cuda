/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates"
 by Jeffrey Mahovsky and Brian Wyvill Department of Computer Science, University
 of Calgary

  This source code is public domain, but please mention us if you use it.

 ******************************************************************************/

#include "JGT-float/smits_mul.h"

namespace jgt_float {

bool smits_mul(ray *r, aabox *b, float *t) {
  float tnear = -1e6;
  float tfar = 1e6;

  {
    // multiply by the inverse instead of dividing
    float t1 = (b->x0 - r->x) * r->ii;
    float t2 = (b->x1 - r->x) * r->ii;

    if (t1 > t2) {
      float temp = t1;
      t1 = t2;
      t2 = temp;
    }
    if (t1 > tnear)
      tnear = t1;
    if (t2 < tfar)
      tfar = t2;

    if (tnear > tfar)
      return false;
    if (tfar < 0.0)
      return false;
  }
  {
    float t1 = (b->y0 - r->y) * r->ij;
    float t2 = (b->y1 - r->y) * r->ij;

    if (t1 > t2) {
      float temp = t1;
      t1 = t2;
      t2 = temp;
    }
    if (t1 > tnear)
      tnear = t1;
    if (t2 < tfar)
      tfar = t2;

    if (tnear > tfar)
      return false;
    if (tfar < 0.0)
      return false;
  }
  {
    float t1 = (b->z0 - r->z) * r->ik;
    float t2 = (b->z1 - r->z) * r->ik;

    if (t1 > t2) {
      float temp = t1;
      t1 = t2;
      t2 = temp;
    }
    if (t1 > tnear)
      tnear = t1;
    if (t2 < tfar)
      tfar = t2;

    if (tnear > tfar)
      return false;
    if (tfar < 0.0)
      return false;
  }

  *t = tnear;
  return true;
}

} // namespace jgt_float
