#pragma once

#include "apbd/Model.h"
#include "se3/lib.h"
#include "util.h"
#include <math.h>

apbd::Model createModelSample(int modelID) {
  auto model = apbd::Model();

  switch (modelID) {

  case 0: {
    // model.name = 'Rigid Collisions';
    // model.plotH = false;
    model.tEnd = 1;
    model.h = 5e-3;
    model.substeps = 1;
    model.iters = 30;
    float density = 1.0;
    float w = 1;
    Eigen::Vector3f sides{w, w, w};
    model.gravity = Eigen::Vector3f(0, 0, -980).transpose();
    model.ground_E = Eigen::Matrix4f::Identity();
    float mu = 0.01;

    model.ground_size = 10;
    model.axis = 5 * Eigen::Matrix<float, 6, 1>(-1, 1, -1, 1, 0, 1);
    // model.drawHz = 10000;

    // model.view = [0 0];

    size_t n = 9;
    model.bodies = (apbd::Body *)alloc_device(n * sizeof(apbd::Body));
    for (size_t i = 0; i < n; i++) {
      model.bodies[i] = apbd::Body(
          apbd::BodyRigid(apbd::ShapeCuboid{sides}, density, true, mu));
      Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
      float x = 0.05 * i;
      float y = 0;
      float z = (i - 0.5) * w * 0.99;
      E.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
      model.bodies[i].setInitTransform(E);
      if (i == 1) {
        model.bodies[i].setInitVelocity(
            Eigen::Matrix<float, 6, 1>(1, 0, 0, 0, 0, 0));
      }
    }
  }

  case 10:
    // model.name = 'Rigid Body';
    // model.plotH = true;
    model.tEnd = 1;
    model.h = 1. / 30.;
    model.substeps = 10;
    model.iters = 1;
    float density = 1.0;
    float l = 5;
    float w = 1;
    auto sides = Eigen::Vector3f(l, w, w);
    model.gravity = 0 * Eigen::Vector3f(0, 0, -980).transpose();
    model.ground_E = Eigen::Matrix4f::Identity();

    model.ground_size = 20;
    model.axis = 10 * Eigen::Matrix<float, 6, 1>(-1, 1, -1, 1, 0, 1);
    // model.drawHz = 30;

    model.body_count = 1;
    model.bodies = (apbd::Body *)alloc_device(1 * sizeof(apbd::Body));
    model.bodies[0] = std::move(apbd::Body{
        apbd::Body(apbd::BodyRigid(apbd::ShapeCuboid{sides}, density))});

    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R = se3::aaToMat(Eigen::Vector3f(1, 1, 1), M_PI / 4);
    E.block<3, 3>(0, 0) = R;
    E.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 5);
    model.bodies[0].setInitTransform(E);
    Eigen::Vector3f x1 = R.transpose() * Eigen::Vector3f(3, -4, 5);
    Eigen::Vector3f x2 = R.transpose() * Eigen::Vector3f(0, 0, 5);
    Eigen::Matrix<float, 6, 1> v;
    v << x1, x2;
    model.bodies[0].setInitVelocity(v);
    // model.bodies{end}.setInitVelocity([0 0 0 0 0 1]');
  }
  model.init();

  return model;
}
