#include "model_samples.h"
#include <exception>
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using std::cout, std::endl, std::string, std::runtime_error;
typedef std::chrono::high_resolution_clock Clock;

__global__ void kernel(apbd::Model model, apbd::Body *body_buffer,
                       apbd::Body **body_ptr_buffer,
                       apbd::Constraint *constraint_buffer, int sims) {
  // get this scene ID
  size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (scene_id >= sims)
    return;
  // make a copy of the model
  apbd::Model thread_model =
      apbd::Model::clone_with_buffers(model, scene_id, body_buffer);

  Eigen::Matrix3f R = se3::aaToMat(
      Eigen::Vector3f(1, 1, 1), static_cast<float>(scene_id) * 0.5 * M_PI / 4);

  for (size_t index = 0; index < thread_model.body_count; index++) {
    auto body = &thread_model.bodies[index];
    body->data.rigid.x.block<4, 1>(0, 0) = Eigen::Quaternionf(R).coeffs();
  }

  // create a thread-local collider
  auto collider = apbd::Collider(&thread_model, scene_id, body_ptr_buffer,
                                 constraint_buffer);
  // simulate
  thread_model.simulate(&collider);
}

void run_kernel(apbd::Model model, int sims) {
  cout << "# thread blocks: " << (sims + BLOCK_SIZE - 1) / BLOCK_SIZE << endl;

  // const size_t shared_size = c.constraints.size() * sizeof(Constraint);
  const size_t shared_size = 0;
  // std::cout << "kernel shared_size: " << shared_size << " = " <<
  // c.constraints.size() << " * " << sizeof(Constraint) << std::endl;

  size_t body_buffer_size = sims * model.body_count;
  apbd::Body *body_buffer = alloc_device<apbd::Body>(body_buffer_size);
  apbd::Body **body_ptr_buffer = nullptr;
  apbd::Constraint *constraint_buffer = nullptr;
  apbd::Collider::allocate_buffers(model, sims, body_ptr_buffer,
                                   constraint_buffer);

  model.move_to_device();

  auto t1 = Clock::now();

  kernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_size>>>(
      model, body_buffer, body_ptr_buffer, constraint_buffer, sims);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  std::cout << "# Kernel took: " << (t2 - t1).count() << '\t';
}

void run_cpu_thread(apbd::Model model, int sims, int processor_count, int id) {
  for (int i = id; i < sims; i += processor_count) {
    auto collider = apbd::Collider(&model);
    model.simulate(&collider);
  }
}

void cpu_run_group(apbd::Model model, int sims) {
  const auto processor_count = std::thread::hardware_concurrency();
  if (processor_count == 0) {
    throw runtime_error("Failed to detect concurrency.");
  }
  auto handles = std::vector<std::thread>();
  auto t1 = Clock::now();
  for (int i = 0; i < processor_count; i++) {
    handles.push_back(
        std::thread(run_cpu_thread, model, sims, processor_count, i));
  }
  for (auto &h : handles) {
    h.join();
  }
  auto t2 = Clock::now();
  cout << "# Kernel took: " << (t2 - t1).count() << '\t';
}

struct MainState {
  bool visualize;
  string output_file;
  int model_id;
  unsigned long scene_count;
};

const char *HELP = "\
arguments:          \n\
  -a, --algorithm ALG  Which version of the simulation algorithm to use\n\
                       values: v, gp; default v\n\
  -o, --output FILE    An output file to write the simulated positions to\n\
  -i, --model_id ID    The model scene to use\n\
  -s, --scene-count N  The number of simulations to run\n\
  -v, --visualize      Whether to visualize the finished simulation. Only\n\
                       supported on some algorithms\n\
  -h, --help           Show this help text\n\
";

MainState parse_arguments(int argc, char *argv[]) {
  struct MainState state = {
      .visualize = false,
      .output_file = "",
      .model_id = 0,
      .scene_count = 1,
  };

  option longopts[] = {{"visualize", no_argument, NULL, 'v'},
                       {"output", required_argument, NULL, 'o'},
                       {"model_id", required_argument, NULL, 'm'},
                       {"scene-count", required_argument, NULL, 's'},
                       // {"algorithm", required_argument, NULL, 'a'},
                       {"help", no_argument, NULL, 'h'},
                       {0}};

  while (1) {
    const int opt = getopt_long(argc, argv, "hvo:m:s:a:", longopts, 0);

    if (opt == -1) {
      break;
    }
    string o;
    switch (opt) {
    case 'h':
      cout << HELP << endl;
      exit(0);
    case 'v':
      state.visualize = true;
      break;
    case 'o':
      if (optarg == NULL) {
        throw runtime_error("argument for output file is required.");
      }
      cout << "# output-file: " << optarg << endl;
      state.output_file = optarg;
      break;
    case 'm':
      if (optarg == NULL) {
        break;
      }
      cout << "# model_id: " << optarg << endl;
      state.model_id = std::stoi(optarg);
      break;
    case 's':
      if (optarg == NULL) {
        break;
      }
      cout << "# scene-count: " << optarg << endl;
      state.scene_count = std::stoul(optarg);
      break;
    // case 'a':
    //   o = optarg;
    //   if (o == "v") {
    //     state.algorithm = VALIDATION;
    //   } else if (o == "gp") {
    //     state.algorithm = GPU_PARALLEL;
    //   } else {
    //     throw runtime_error("Unknown algorithm: " + string(optarg) +
    //                         ", try v or gp.");
    //   }
    //   break;
    case '?':
    default:
      cout << "unknown option." << endl;
    }
  }
  return state;
}

int main(int argc, char *argv[]) {
  auto state = parse_arguments(argc, argv);
  auto model = createModelSample(state.model_id);

  auto t1 = Clock::now();
#ifdef USE_CUDA
  cout << "# Running with CUDA" << endl;
  run_kernel(model, state.scene_count);
#else
  cout << "# Running on CPU" << endl;
  cpu_run_group(model, state.scene_count);
#endif
  auto t2 = Clock::now();
  cout << " Simulation took: " << (t2 - t1).count() << '\n';
}
