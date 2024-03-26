#include "apbd/Body.h"
#include "se3/lib.h"
#include <limits>
#include <svd3_cuda.h>

namespace apbd {

// convenience layer between eigen matrices and svd3_cuda
__host__ __device__ void svd_step(Eigen::Matrix3f &A) {
  Eigen::Matrix3f U, S, V;

  svd(
      // input A
      A(0), A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8),
      // output U
      U(0), U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8),
      // output S
      S(0), S(1), S(2), S(3), S(4), S(5), S(6), S(7), S(8),
      // output V
      V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7), V(8));

  A = U * V.transpose();
}

using Eigen::Vector3f;

Body::Body() : type(BODY_INVALID), data{0} {}
Body::Body(BodyRigid rigid) : type(BODY_RIGID), data{.rigid{rigid}} {}
Body::Body(BodyAffine affine)
    : type(BODY_AFFINE), data{.affine = std::move(affine)} {}
Body &Body::operator=(const Body &&other) {
  this->type = other.type;
  switch (type) {
  case BODY_AFFINE:
    this->data.affine = std::move(other.data.affine);
    break;
  case BODY_RIGID:
    this->data.rigid = std::move(other.data.rigid);
    break;
  default:
    break;
  }
  return *this;
}

void Body::init() {
  switch (this->type) {
  case BODY_AFFINE: {
    // auto &data = this->data.affine;
    // data.computeInertiaConst();
    // data.x = data.xInit;
    // data.x0 = data.x;
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.computeInertiaConst();
    data.x = data.xInit;
    data.x0 = data.x;
    break;
  }
  default:
    break;
  }
}

void Body::stepBDF1(unsigned int step, unsigned int substep, float hs,
                    Eigen::Vector3f gravity) {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    auto xdot = data.computeVelocity(step, substep, hs);
    data.x0 = data.x;
    Eigen::Vector3f axdot = xdot.block<3, 1>(0, 0);
    Eigen::Vector3f aydot = xdot.block<3, 1>(3, 0);
    Eigen::Vector3f azdot = xdot.block<3, 1>(6, 0);
    Eigen::Vector3f pdot = xdot.block<3, 1>(9, 0);
    auto w = data.Wp;
    auto W = data.Wa;
    Vector3f f = Eigen::Vector3f::Zero();
    auto t = Eigen::Matrix<float, 9, 1>::Zero(); // affine torque
    Eigen::Vector3f tx = t.block<3, 1>(0, 0);
    Eigen::Vector3f ty = t.block<3, 1>(3, 0);
    Eigen::Vector3f tz = t.block<3, 1>(6, 0);
    f = f + gravity / w; // Gravity
    // Integrate velocities
    axdot = axdot + hs * Eigen::Vector3f(W.array() * tx.array());
    aydot = aydot + hs * Eigen::Vector3f(W.array() * ty.array());
    azdot = azdot + hs * Eigen::Vector3f(W.array() * tz.array());
    pdot = pdot + hs * (w * f);
    // Integrate positions
    data.x.block<3, 1>(0, 0) = data.x.block<3, 1>(0, 0) + hs * axdot;
    data.x.block<3, 1>(3, 0) = data.x.block<3, 1>(3, 0) + hs * aydot;
    data.x.block<3, 1>(6, 0) = data.x.block<3, 1>(6, 0) + hs * azdot;
    data.x.block<3, 1>(9, 0) = data.x.block<3, 1>(9, 0) + hs * pdot;

    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    auto xdot = data.computeVelocity(step, substep, hs);
    Eigen::Vector4f qdot = xdot.block<4, 1>(0, 0);
    Eigen::Vector3f v = xdot.block<3, 1>(4, 0); // pdot
    data.x0 = data.x;
    Eigen::Vector4f q = data.x.block<4, 1>(0, 0);
    Eigen::Vector3f p = data.x.block<3, 1>(4, 0);
    auto w = se3::qdotToW(q, qdot); // angular velocity in body coords
    Vector3f f = Vector3f::Zero();  // translational force in world space
    Vector3f t = Vector3f::Zero();  // angular torque in body space
    auto m = data.Mp;               // scalar mass
    auto I = data.Mr;               // inertia in body space
    Eigen::Vector3f Iw =
        I.array() * w.array(); // angular momentum in body space
    f = f + m * gravity;       // Gravity
    t = t + se3::cross(Iw, w); // Coriolis
    // Integrate velocities
    w = w + hs * Eigen::Vector3f(t.array() / I.array());
    v = v + hs * (f / m);
    qdot = se3::wToQdot(q, w);
    // Integrate positions
    q = q + hs * qdot;
    p = p + hs * v;
    q = q / q.norm();
    data.x.block<4, 1>(0, 0) = q;
    data.x.block<3, 1>(4, 0) = p;
    data.x1_0 = data.x;
    data.x1 = data.x1_0;
    break;
  }
  default:
    break;
  }
}

constexpr unsigned int UNSIGNED_MAX = std::numeric_limits<unsigned int>::max();

void Body::clearShock() {
  // set layer to MAX
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.layer = UNSIGNED_MAX;
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.layer = UNSIGNED_MAX;
    break;
  }
  default:
    break;
  }
}

void Body::applyJacobiShock() {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.x1 += data.dxJacobiShock;
    data.dxJacobiShock = vec12::Zero();
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.x1 += data.dxJacobiShock;
    data.dxJacobiShock = vec7::Zero();
    break;
  }
  default:
    break;
  }
  this->regularize();
}

void Body::regularize() {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.x = data.x1;
    Eigen::Matrix3f A = data.x.reshaped(3, 3);
    // Eigen::JacobiSVD<Eigen::Matrix3f> svd(A);
    // A = svd.matrixU() * svd.matrixV().transpose();
    svd_step(A);
    data.x = A.reshaped(9, 1);
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.regularize();
    break;
  }
  default:
    break;
  }
}

void Body::setInitTransform(Eigen::Matrix4f E) {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.xInit.block<9, 1>(0, 0) = E.block<3, 3>(0, 0).reshaped(9, 1);
    data.xInit.block<3, 1>(9, 0) = E.block<3, 1>(0, 3);
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.xInit.block<4, 1>(0, 0) =
        Eigen::Quaternionf(E.block<3, 3>(0, 0)).coeffs();
    if (data.xInit(3) < 0) {
      data.xInit = -data.xInit;
    }
    data.xInit.block<3, 1>(4, 0) = E.block<3, 1>(0, 3);
    break;
  }
  default:
    break;
  }
}

void Body::setInitVelocity(Eigen::Matrix<float, 6, 1> velocity) {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    auto E = data.computeInitTransform();
    Eigen::Matrix4f Edot = E * se3::brac(velocity);
    data.xdotInit.block<9, 1>(0, 0) = Edot.block<3, 3>(0, 0).reshaped(9, 1);
    data.xdotInit.block<3, 1>(9, 0) = Edot.block<3, 1>(0, 3);
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    Eigen::Quaternionf q = Eigen::Quaternionf(data.xInit.block<4, 1>(0, 0));
    data.xdotInit.block<3, 1>(4, 0) = (q * velocity.block<3, 1>(3, 0));
    data.xdotInit.block<4, 1>(0, 0) =
        se3::wToQdot(q.coeffs(), velocity.block<3, 1>(0, 0));
    break;
  }
  default:
    break;
  }
}

Eigen::Matrix4f Body::computeTransform() {
  switch (this->type) {
  case BODY_AFFINE: {
    // auto &data = this->data.affine;
    // TODO
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    return data.computeTransform();
  }
  default:
    break;
  }
}

bool Body::broadphaseGround(Eigen::Matrix4f Eg) {
  switch (this->type) {
  case BODY_AFFINE: {
    // auto &data = this->data.affine;
    // TODO
    return false;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    Eigen::Matrix4f E = data.computeTransform();
    return data.shape.broadphaseGround(E, Eg);
  }
  default:
    return false;
  }
}

cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>
Body::narrowphaseGround(Eigen::Matrix4f Eg) {
  switch (this->type) {
  case BODY_AFFINE: {
    // auto &data = this->data.affine;
    // TODO
    return cuda::std::pair(cuda::std::array<CollisionGround, 8>(), 0);
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    Eigen::Matrix4f E = data.computeTransform();
    return data.shape.narrowphaseGround(E, Eg);
  }
  default:
    return cuda::std::pair(cuda::std::array<CollisionGround, 8>(), 0);
  }
}

bool Body::broadphaseRigid(Body *other) {
  switch (this->type) {
  case BODY_AFFINE: {
    // auto &data = this->data.affine;
    // TODO
    return false;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    Eigen::Matrix4f E1 = data.computeTransform();
    Eigen::Matrix4f E2 = other->computeTransform();
    // TODO: handle shape for affine
    return data.shape.broadphaseShape(E1, &other->data.rigid.shape, E2);
  }
  default:
    break;
  }
}

cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>
Body::narrowphaseRigid(Body *other) {
  switch (this->type) {
  case BODY_AFFINE: {
    // auto &data = this->data.affine;
    // TODO
    return cuda::std::pair(cuda::std::array<CollisionRigid, 8>(), 0);
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    Eigen::Matrix4f E1 = data.computeTransform();
    Eigen::Matrix4f E2 = other->computeTransform();
    // TODO: handle shape for affine
    return data.shape.narrowphaseShape(E1, &other->data.rigid.shape, E2);
  }
  default:
    return cuda::std::pair(cuda::std::array<CollisionRigid, 8>(), 0);
  }
}

bool Body::collide() {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    return data.collide;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    return data.collide;
  }
  default:
    return false;
  }
}

BodyRigid::BodyRigid(Shape shape, float density)
    : xInit(vec7::Zero()), xdotInit(vec7::Zero()), x(vec7::Zero()),
      x0(vec7::Zero()), x1(vec7::Zero()), x1_0(vec7::Zero()),
      dxJacobi(vec7::Zero()), dxJacobiShock(vec7::Zero()), collide(false),
      mu(0.0), layer(99), shape(shape), density(density),
      Mr(Eigen::Vector3f::Zero()), Mp(0) {}
BodyRigid::BodyRigid(Shape shape, float density, bool collide, float mu)
    : xInit(vec7::Zero()), xdotInit(vec7::Zero()), x(vec7::Zero()),
      x0(vec7::Zero()), x1(vec7::Zero()), x1_0(vec7::Zero()),
      dxJacobi(vec7::Zero()), dxJacobiShock(vec7::Zero()), collide(collide),
      mu(mu), layer(99), shape(shape), density(density),
      Mr(Eigen::Vector3f::Zero()), Mp(0) {}

vec7 BodyRigid::computeVelocity(unsigned int step, unsigned int substep,
                                float hs) {
  if (step == 0 && substep == 0)
    return this->xdotInit;
  else
    return (this->x - this->x0) / hs;
}

void BodyRigid::computeInertiaConst() {
  auto I = this->shape.computeInertia(this->density);
  this->Mr = I.block<3, 1>(0, 0);
  this->Mp = I(4);
}

Eigen::Vector3f BodyRigid::computePointVel(Eigen::Vector3f xl, float hs) {
  // xdot = this.computeVelocity(k,ks,hs);
  vec7 xdot = (this->x - this->x0) / hs;
  Eigen::Vector4f qdot = xdot.block<4, 1>(0, 0);
  Eigen::Vector3f pdot = xdot.block<3, 1>(4, 0); // in world coords
  Eigen::Quaternionf q = Eigen::Quaternionf(this->x.block<4, 1>(0, 0));
  Eigen::Vector3f w =
      se3::qdotToW(q.coeffs(), qdot); // angular velocity in body coords
  // v = R*cross(w,xl) + pdot
  return (q * w.cross(xl)) + pdot;
}

Eigen::Matrix4f BodyRigid::computeTransform() {
  Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
  E.block<3, 3>(0, 0) =
      Eigen::Quaternionf(x.block<4, 1>(0, 0)).toRotationMatrix();
  E.block<3, 1>(0, 3) = x.block<3, 1>(4, 0);
  return E;
}

void BodyRigid::applyJacobi() {
  this->x1 += this->dxJacobi;
  this->dxJacobi = vec7::Zero();
  this->regularize();
}

void BodyRigid::regularize() {
  this->x = this->x1;
  Eigen::Vector4f q = this->x.block<4, 1>(0, 0);
  q /= q.norm();
  this->x.block<4, 1>(0, 0) = q;
}

vec12 BodyAffine::computeVelocity(unsigned int step, unsigned int substep,
                                  float hs) {
  // TODO
}

Eigen::Matrix4f BodyAffine::computeInitTransform() {
  // TODO
}

void Body::write_state() {
  Vector3f position;
  Eigen::Vector4f rotation;
  switch (this->type) {
  case BODY_AFFINE: {
    return;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    position = data.x.block<3, 1>(4, 0);
    rotation = data.x.block<4, 1>(0, 0);
    break;
  }
  default:
    return;
  }
  printf("%f %f %f r %f %f %f %f", position(0), position(1), position(2),
         rotation(0), rotation(1), rotation(2), rotation(3));
}

} // namespace apbd
