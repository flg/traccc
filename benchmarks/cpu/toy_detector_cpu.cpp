/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Traccc core include(s).
#include "traccc/geometry/detector.hpp"

// Traccc algorithm include(s).
#include "traccc/finding/ckf_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Traccc IO include(s).
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"

// Local include(s).
#include "benchmarks/toy_detector_benchmark.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Google benchmark include(s).
#include <benchmark/benchmark.h>

BENCHMARK_F(ToyDetectorBenchmark, CPU)(benchmark::State& state) {

    // Type declarations
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t,
                           typename detector_type::algebra_type,
                           detray::constrained_step<>>;
    using host_detector_type = traccc::default_detector::host;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;

    // Read back detector file
    host_detector_type det{host_mr};
    traccc::io::read_detector(
        det, host_mr, sim_dir + "toy_detector_geometry.json",
        sim_dir + "toy_detector_homogeneous_material.json",
        sim_dir + "toy_detector_surface_grids.json");

    // B field
    auto field = detray::bfield::create_const_field(B);

    // Algorithms
    traccc::seeding_algorithm sa(seeding_cfg, grid_cfg, filter_cfg, host_mr);
    traccc::track_params_estimation tp(host_mr);
    traccc::host::ckf_algorithm host_finding(finding_cfg);
    traccc::fitting_algorithm<host_fitter_type> host_fitting(fitting_cfg);

    for (auto _ : state) {

// Iterate over events
#pragma omp parallel for
        for (unsigned int i_evt = 0; i_evt < n_events; i_evt++) {

            auto& spacepoints_per_event = spacepoints[i_evt];
            auto& measurements_per_event = measurements[i_evt];

            // Seeding
            auto seeds = sa(spacepoints_per_event);

            // Track param estimation
            auto params = tp(spacepoints_per_event, seeds, B);

            // Track finding with CKF
            auto track_candidates = host_finding(
                det, field, vecmem::get_data(measurements_per_event),
                vecmem::get_data(params));

            // Track fitting with KF
            auto track_states = host_fitting(det, field, track_candidates);
        }
    }

    state.counters["event_throughput_Hz"] = benchmark::Counter(
        static_cast<double>(n_events), benchmark::Counter::kIsRate);
}
