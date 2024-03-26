#include "apbd/Model.h"
#include "util.h"

#include <iostream>
#include <stdexcept>

namespace apbd {

Model::Model()
    : h(1. / 30.), tEnd(1), substeps(10), bodies(nullptr), body_count(0),
      constraints(nullptr), constraint_count(0), constraint_layers(nullptr),
      layer_count(0), constraint_layer_sizes(nullptr), body_layers(nullptr),
      body_layer_sizes(nullptr), gravity(0.0, 0.0, -980.0), iters(1),
      ground_E(Eigen::Matrix4f::Zero()), ground_size(10),
      axis(Eigen::Matrix<float, 6, 1>::Zero()), steps(0) {}

Model::Model(const Model &other)
    : h(other.h), tEnd(other.tEnd), substeps(other.substeps), bodies(nullptr),
      body_count(other.body_count), constraints(other.constraints),
      constraint_count(other.constraint_count), constraint_layers(nullptr),
      layer_count(other.layer_count), constraint_layer_sizes(nullptr),
      body_layers(nullptr), body_layer_sizes(nullptr), gravity(other.gravity),
      iters(other.iters), ground_E(other.ground_E),
      ground_size(other.ground_size), axis(other.axis), steps(other.steps) {
#ifdef __CUDA_ARCH__
  // we are on the device; don't copy the bodies
#else
  cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, other.bodies));
  if (attributes.type == cudaMemoryTypeDevice) {
    bodies = alloc_device<Body>(other.body_count);
    memcpy_device(bodies, other.bodies, other.body_count);
  } else {
    size_t size = other.body_count * sizeof(Body);
    bodies = new Body[other.body_count];
    memcpy(bodies, other.bodies, size);
  }
#endif
}

void Model::init() {
  // initialize bodies
  // if (this->bodies == nullptr) {
  //   throw std::runtime_error("Bodies not initialized");
  // }
  for (size_t i = 0; i < this->body_count; i++) {
    this->bodies[i].init();
  }
  // create constraints
  // if (this->constraints == nullptr && this->constraint_count) {
  //   throw std::runtime_error("Constraints not initialized");
  // }
  for (size_t i = 0; i < this->constraint_count; i++) {
    this->constraints[i].init();
  }
  // calculate parameters
  this->steps = ceil(this->tEnd / this->h);
  // this->hs = this->h / this->substeps;
  // this->k = 0;
  // this->ks
  this->print_config();
  this->write_state(0);
}

void Model::move_to_device() {
  bodies = move_array_to_device(bodies, body_count);
  constraints = move_array_to_device(constraints, constraint_count);
  // TODO: layers
}

__device__ Model Model::clone_with_buffers(const Model &other, size_t scene_id,
                                           Body *body_buffer) {
  Model new_model(other);

  new_model.bodies = &body_buffer[other.body_count * scene_id];
  memcpy_device(new_model.bodies, other.bodies, other.body_count);
  return new_model;
}

void Model::simulate(Collider *collider) {
  float time = 0;
  float hs = this->h / static_cast<float>(this->substeps);
  for (unsigned int step = 0; step < this->steps; step++) {
    this->clearBodyShockPropInfo();
    collider->run(this);
    this->constructConstraintGraph();
    for (unsigned int substep = 0; substep < this->substeps; substep++) {
      this->stepBDF1(step, substep, hs);
      this->solveConSP(collider, hs);
      this->solveConGS(collider, hs);
      time += hs;
    }
    this->computeEnergies();
    // draw ?
    this->write_state(step + 1);
  }
}

/** Private Functions **/

void Model::stepBDF1(unsigned int step, unsigned int substep, float hs) {
  for (size_t body_i = 0; body_i < this->body_count; body_i++) {
    this->bodies[body_i].stepBDF1(step, substep, hs, this->gravity);
  }
}
void Model::clearBodyShockPropInfo() {
  // TODO
  // clears the shock propagation info from each body; this may not be necessary
  // depending on implementation
  for (size_t body_i = 0; body_i < this->body_count; body_i++) {
    this->bodies[body_i].clearShock();
  }
}
void Model::constructConstraintGraph() {
  // TODO
  // Constructs a graph of constraints, working from the ground layer up
  // needs a list of constraints and bodies
  // constraints needs:
  //  - list of bodies
  // body needs:
  //  - layer
  //  - shock parent constraint
  //  - constraint
  //
  //  collect static constraints and collision constraints
  //  for each constraint:
  //    if it is ground, initialize the body affected to layer 1 and add this
  //    constraint to the list affecting that body if the constraint has 2
  //    bodies (not 1), then tell both bodies that this constraint affects them
  //
  //  working up one layer at a time:
  //    for every body affected in the previous layer:
  //      for each constraint affecting the body:
  //        make sure the body on a higher layer is second
  //        assign the second body to this layer, and add this constraint to the
  //        parent constraints add the second body and this constraint to this
  //        layer

  // in theory, we can do this by keeping a constraint and body layer list, and
  // a last layer constraint list. the shock parent list is more difficult, but
  // does not seem to be used for anything other than setting shockProp to true;
  // so we can do this here
}
void Model::solveConSP(Collider *collider, float hs) {
  for (size_t constraint_i = 0; constraint_i < this->constraint_count;
       constraint_i++) {
    this->constraints[constraint_i].clear();
  }

  // TODO: remove and add graph creation
  for (size_t i = 0; i < collider->collision_count; i++) {
    collider->collisions[i].solve(hs, true);
  }
  for (size_t i = 0; i < this->constraint_count; i++) {
    this->constraints[i].solve(hs, true);
  }
  for (size_t i = 0; i < this->body_count; i++) {
    this->bodies[i].applyJacobiShock();
  }
  for (size_t i = 0; i < collider->collision_count; i++) {
    collider->collisions[i].solve(hs, true);
  }
  for (size_t i = 0; i < this->constraint_count; i++) {
    this->constraints[i].solve(hs, true);
  }

  for (size_t i = 0; i < this->layer_count; i++) {
    for (int iter = 0; iter < this->iters; iter++) {
      for (size_t j = 0; j < this->constraint_layer_sizes[i]; j++) {
        this->constraints[this->constraint_layers[i * MAX_LAYER_SIZE + j]]
            .solve(hs, true);
      }
    }
  }

  for (long i = this->layer_count - 1; i >= 0; i--) {
    for (size_t j = 0; j < this->body_layer_sizes[i]; j++) {
      this->bodies[this->body_layers[i * MAX_LAYER_SIZE + j]]
          .applyJacobiShock();
    }
    for (int iter = 0; iter < this->iters; iter++) {
      for (size_t j = 0; j < this->constraint_layer_sizes[i]; j++) {
        this->constraints[this->constraint_layers[i * MAX_LAYER_SIZE + j]]
            .solve(hs, true);
      }
    }
  }
}
void Model::solveConGS(Collider *collider, float hs) {
  for (int iter = 0; iter < this->iters; iter++) {
    for (size_t constraint_i = 0; constraint_i < this->constraint_count;
         constraint_i++) {
      this->constraints[constraint_i].solve(hs, false);
    }
    for (size_t i = 0; i < collider->collision_count; i++) {
      collider->collisions[i].solve(hs, false);
    }
  }
}
void Model::computeEnergies() { /*TODO*/
}

void Model::write_state(unsigned int step) {
#ifdef WRITE
#ifdef __CUDA_ARCH__
  if (threadIdx.x == 0)
    printf("Step %d\n", step);
  // print up to 8 simulations in parallel
  for (size_t i = 0; i < body_count * 8; i++) {
    if (i / 9 != threadIdx.x)
      continue;
    printf("%lu ", i);
    bodies[i % 9].write_state();
    printf("\n");
  }
#else
  printf("Step %d\n", step);
  for (size_t i = 0; i < body_count; i++) {
    printf("%lu ", i);
    bodies[i].write_state();
    printf("\n");
  }
#endif
#endif
}

void Model::print_config() {
  printf("# Body count: %lu\n"
         "# Constraint count: %lu\n"
         "# Gravity: [%f %f %f]\n"
         "# Ground size: %f\n"
         "# Time Step: %f\n"
         "# End Time: %f\n"
         "# Steps: %u\n"
         "# Substeps: %u\n"
         "# Iterations: %u\n",
         body_count, constraint_count, gravity(0), gravity(1), gravity(2),
         ground_size, h, tEnd, steps, substeps, iters);
}

} // namespace apbd
