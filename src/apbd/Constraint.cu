#include <apbd/Collider.h>
#include <apbd/Constraint.h>
#include <se3/lib.h>

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f, Eigen::Quaternionf;

namespace apbd {

Constraint::Constraint(ConstraintRigid rigid)
    : type(CONSTRAINT_COLLISION_RIGID), data{.rigid = {rigid}} {}
Constraint::Constraint(ConstraintGround ground)
    : type(CONSTRAINT_COLLISION_GROUND), data{.ground = {ground}} {}
Constraint::Constraint(ConstraintJointRevolve revolve)
    : type(CONSTRAINT_JOINT_REVOLVE), data{.joint_revolve = {revolve}} {}

ConstraintGround::ConstraintGround(BodyRigid *body, Eigen::Matrix4f Eg, float d,
                                   Eigen::Vector3f xl, Eigen::Vector3f xw,
                                   Eigen::Vector3f nw, Eigen::Vector3f vw)
    : C(Vector3f::Zero()), lambda(Vector3f::Zero()), nw(nw),
      lambdaSF(Vector3f::Zero()), d(0), dlambdaNor(0), shockProp(false),
      body(body), Eg(Eg), xl(xl), xw(xw), vw(vw) {}

ConstraintRigid::ConstraintRigid(BodyRigid *body1, BodyRigid *body2, float d,
                                 Eigen::Vector3f nw, Eigen::Vector3f x1,
                                 Eigen::Vector3f x2)
    : C(Vector3f::Zero()), lambda(Vector3f::Zero()), nw(nw),
      lambdaSF(Vector3f::Zero()), d(0), dlambdaNor(0), shockProp(false),
      body1(body1), body2(body2), x1(x1), x2(x2) {}

Constraint &Constraint::operator=(const Constraint &other) {
  this->type = other.type;
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    this->data.ground = other.data.ground;
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    this->data.rigid = other.data.rigid;
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    this->data.joint_revolve = other.data.joint_revolve;
    break;
  }

  default:
    break;
  }
  return *this;
}

void Constraint::clear() {
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    ConstraintGround *c = &data.ground;
    c->C = Eigen::Vector3f::Zero();
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    ConstraintRigid *c = &data.rigid;
    c->C = Eigen::Vector3f::Zero();
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    ConstraintJointRevolve *c = &data.joint_revolve;
    c->C = Eigen::Vector3f::Zero();
    c->lambda = Eigen::Vector3f::Zero();
    break;
  }

  default:
    break;
  }
}

void Constraint::solve(float hs, bool doShockProp) {
  switch (type) {
  case CONSTRAINT_COLLISION_GROUND: {
    ConstraintGround *c = &data.ground;
    c->solveNorPos(hs);
    c->applyJacobi();
    break;
  }
  case CONSTRAINT_COLLISION_RIGID: {
    ConstraintRigid *c = &data.rigid;
    c->solveNorPos(hs);
    c->applyJacobi();
    break;
  }
  case CONSTRAINT_JOINT_REVOLVE: {
    ConstraintJointRevolve *c = &data.joint_revolve;
    c->solve();
    break;
  }

  default:
    break;
  }
}

void ConstraintGround::solveNorPos(float hs) {
  Vector3f v = hs * body->computePointVel(xl, hs);
  float vNorm = v.norm();
  Vector3f vNormalized = v / vNorm;
  Vector3f tx = Eg.block<3, 1>(0, 0);
  Vector3f ty = Eg.block<3, 1>(0, 1);
  Eigen::Matrix3f frame_tmp;
  frame_tmp << nw, tx, ty;
  Vector3f vNormalizedContactFrame = frame_tmp * vNormalized;

  float dlambda = solvePosDir1(vNorm, vNormalized);
  C = vNorm * vNormalizedContactFrame;

  float dlambdaNor = dlambda * vNormalizedContactFrame(0);
  float lambdaNor = lambda(1) + dlambdaNor;
  if (lambdaNor < 0) {
    dlambdaNor = -lambda(1);
  }
  lambda(1) += dlambdaNor;
  float mu = body->mu;
  Vector3f dlambdaTan;
  if (mu > 0) {
    float dlambdaTx = dlambda * vNormalizedContactFrame(1);
    float dlambdaTy = dlambda * vNormalizedContactFrame(2);
    float lambdaNorLenMu = mu * lambda(1);
    Vector2f lambdaTan = Vector2f(lambda(2) + dlambdaTx, lambda(3) + dlambdaTy);
    float lambdaTanLen = lambdaTan.norm();
    auto dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
    if (lambdaTanLen > lambdaNorLenMu) {
      dlambdaTan = (lambdaTan / lambdaTanLen * lambdaNorLenMu -
                    Vector2f(lambda(2), lambda(3)));
    }
    lambda(2) += dlambdaTan(0);
    lambda(3) += dlambdaTan(1);
  }

  Vector3f frictionalContactLambda = Vector3f(dlambdaNor, 0, 0) + dlambdaTan;
  dlambda = frictionalContactLambda.norm();
  if (dlambda > 0) {
    // frictionalContactNormal = [this->nw, tx, ty] * frictionalContactLambda ./
    // dlambda;
    Eigen::Matrix3f tmp;
    tmp << nw, tx, ty;
    Vector3f frictionalContactNormal =
        (tmp * frictionalContactLambda).array() / dlambda;
    vec7 dq = computeDx(dlambda, frictionalContactNormal);
    body->dxJacobi.block<4, 1>(0, 0) += dq.block<4, 1>(0, 0);
    body->dxJacobi.block<3, 1>(4, 0) += dq.block<3, 1>(4, 0);
  }
}

float ConstraintGround::solvePosDir1(float c, Eigen::Vector3f nw) {
  // Use the provided normal rather than normalizing
  auto m1 = this->body->Mp;
  auto I1 = this->body->Mr;
  Quaternionf q1 = Quaternionf(this->body->x.block<4, 1>(0, 0));
  Vector3f nl1 = se3::invert_q(q1) * nw;
  Vector3f rl1 = this->xl;
  Vector3f rnl1 = rl1.cross(nl1);
  float w1 = (1 / m1) + rnl1.transpose() * Vector3f(rnl1.array() / I1.array());
  float numerator = -c;
  float denominator = w1;
  return numerator / denominator;
}

vec7 ConstraintGround::computeDx(float dlambda,
                                 Eigen::Vector3f frictionalContactNormal) {

  float m1 = body->Mp;
  Vector3f I1 = body->Mr;
  // Position update
  Vector3f dpw = dlambda * nw;
  Vector3f dp = dpw / m1;
  // Quaternion update
  Quaternionf q1 = Quaternionf(Eigen::Vector4f(body->x1_0.block<1, 4>(0, 0)));
  auto dpl1 = (se3::invert_q(q1) * dpw);
  Vector4f q2vec;
  q2vec << (q1 * (xl.cross(dpl1).array() / I1.array())), 0;
  // qtmp1 = [I1.\se3.cross(rl1,dpl1); 0];
  // dq = se3.qMul(sin(0.5*qtmp1),q1);
  Quaternionf q2(q2vec);
  Vector4f dq = 0.5 * (q2 * q1).coeffs();
  vec7 out;
  out << dq, dp;
  return out;
}

void ConstraintGround::applyJacobi() { body->applyJacobi(); }
void ConstraintRigid::applyJacobi() {
  body1->applyJacobi();
  body2->applyJacobi();
}

void ConstraintRigid::solveNorPos(float hs) {

  Vector3f v1w = this->body1->computePointVel(this->x1, hs);
  Vector3f v2w = this->body2->computePointVel(this->x2, hs);
  Vector3f v = hs * (v1w - v2w);
  float vNorm = v.norm();
  Vector3f vNormalized = v / vNorm;
  Vector3f tx, ty;
  std::tie(tx, ty) = Collider::generateTangents(this->nw);
  // vNormalizedContactFrame = [-this->nw'; tx' ; ty'] * vNormalized;
  Eigen::Matrix3f tmp;
  tmp << -this->nw, tx, ty;
  Vector3f vNormalizedContactFrame = tmp * vNormalized;

  float dlambda = this->solvePosDir2(v.norm(), vNormalized);
  this->C = vNorm * vNormalizedContactFrame;

  float dlambdaNor = dlambda * vNormalizedContactFrame(1);
  float lambdaNor = this->lambda(1) + dlambdaNor;
  if (lambdaNor < 0) {
    dlambdaNor = -this->lambda(1);
  }
  this->lambda(1) = this->lambda(1) + dlambdaNor;
  float mu1 = this->body1->mu;
  float mu2 = this->body2->mu;
  float mu = 0.5 * (mu1 + mu2);
  Vector2f dlambdaTan{0, 0};
  if (mu > 0) {
    float dlambdaTx = dlambda * vNormalizedContactFrame(2);
    float dlambdaTy = dlambda * vNormalizedContactFrame(3);
    float lambdaNorLenMu = mu * this->lambda(1);
    Vector2f lambdaTan{this->lambda(2) + dlambdaTx,
                       this->lambda(3) + dlambdaTy};
    float lambdaTanLen = lambdaTan.norm();
    dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
    if (lambdaTanLen > lambdaNorLenMu) {
      dlambdaTan = lambdaTan / lambdaTanLen * lambdaNorLenMu -
                   Vector2f(this->lambda(2), this->lambda(3));
    }
    this->lambda(2) = this->lambda(2) + dlambdaTan(1);
    this->lambda(3) = this->lambda(3) + dlambdaTan(2);
  }

  Vector3f frictionalContactLambda;
  frictionalContactLambda << dlambdaNor, dlambdaTan;
  dlambda = frictionalContactLambda.norm();
  if (dlambda > 0) {
    Eigen::Matrix3f tmp;
    tmp << -this->nw, tx, ty;
    Vector3f frictionalContactNormal =
        tmp * Vector3f(dlambda / frictionalContactLambda.array());
    Vector4f dq1, dq2;
    Vector3f dp1, dp2;
    std::tie(dq1, dp1, dq2, dp2) =
        this->computeDx(dlambda, frictionalContactNormal);
    if (this->shockProp) {
      this->body1->dxJacobiShock.block<4, 1>(0, 0) += dq1;
      this->body1->dxJacobiShock.block<3, 1>(4, 0) += dp1;
    } else {
      this->body1->dxJacobi.block<4, 1>(0, 0) += dq1;
      this->body1->dxJacobi.block<3, 1>(4, 0) += dp1;
    }
    this->body2->dxJacobi.block<4, 1>(0, 0) += dq2;
    this->body2->dxJacobi.block<3, 1>(4, 0) += dp2;
  }
}
float ConstraintRigid::solvePosDir2(float c, Eigen::Vector3f nw) {
  // Use the provided normal rather than normalizing
  auto m1 = this->body1->Mp;
  auto m2 = this->body2->Mp;
  auto I1 = this->body1->Mr;
  auto I2 = this->body2->Mr;
  Quaternionf q1 = Quaternionf(this->body1->x.block<4, 1>(0, 0));
  Quaternionf q2 = Quaternionf(this->body2->x.block<4, 1>(0, 0));
  Vector3f nl1 = se3::invert_q(q1) * nw;
  Vector3f nl2 = se3::invert_q(q2) * nw;
  Vector3f rl1 = this->x1;
  Vector3f rl2 = this->x2;
  Vector3f rnl1 = rl1.cross(nl1);
  Vector3f rnl2 = rl2.cross(nl2);
  float w1 = (1 / m1) + rnl1.transpose() * Vector3f(rnl1.array() / I1.array());
  float w2 = (1 / m2) + rnl2.transpose() * Vector3f(rnl2.array() / I2.array());
  float numerator = -c;
  float denominator = w1 + w2;
  return numerator / denominator;
}

std::tuple<Vector4f, Vector3f, Vector4f, Vector3f>
ConstraintRigid::computeDx(float dlambda, Eigen::Vector3f nw) {
  auto m1 = this->body1->Mp;
  auto m2 = this->body2->Mp;
  auto I1 = this->body1->Mr;
  auto I2 = this->body2->Mr;
  // Position update
  Vector3f dpw = dlambda * nw;
  Vector3f dp1 = dpw / m1;
  Vector3f dp2 = -dpw / m2;
  // Quaternion update
  Quaternionf q1 = Quaternionf(this->body1->x1_0.block<4, 1>(0, 0));
  Quaternionf q2 = Quaternionf(this->body2->x1_0.block<4, 1>(0, 0));
  Vector3f dpl1 = se3::invert_q(q1) * dpw;
  Vector3f dpl2 = se3::invert_q(q2) * dpw;
  // qtmp1 = [se3.qRot(q1,I1.\se3.cross(this.x1,dpl1)); 0];
  Vector3f tmp = q1 * (this->x1.cross(dpl1).array() / I1.array());
  Quaternionf qtmp1(0, tmp.x(), tmp.y(), tmp.z());
  // qtmp2 = [se3.qRot(q2,I2.\se3.cross(this.x2,dpl2)); 0];
  Vector3f tmp1 = q2 * (this->x2.cross(dpl2).array() / I2.array());
  Quaternionf qtmp2(0, tmp1.x(), tmp1.y(), tmp1.z());
  // dq1 = se3.qMul(sin(0.5*qtmp1),q1);
  // dq2 = se3.qMul(sin(-0.5*qtmp2),q2);
  Vector4f dq1 = 0.5 * (qtmp1 * q1).coeffs();
  Vector4f dq2 = -0.5 * (qtmp2 * q2).coeffs();
  return std::make_tuple(dq1, dp1, dq2, dp2);
}

void ConstraintJointRevolve::solve() {
  // TODO
}
} // namespace apbd
