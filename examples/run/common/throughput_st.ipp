/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "make_magnetic_field.hpp"

// Project include(s)
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/host_detector.hpp"
#include "traccc/seeding/detail/track_params_estimation_config.hpp"

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/logging.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_gbts_seeding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"

// I/O include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/utils.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// Indicators include(s).
#include <indicators/progress_bar.hpp>

// System include(s).
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>

namespace traccc {

template <typename FULL_CHAIN_ALG>
int throughput_st(std::string_view description, int argc, char* argv[]) {

    std::unique_ptr<const traccc::Logger> prelogger = traccc::getDefaultLogger(
        "ThroughputExample", traccc::Logging::Level::INFO);

    // Program options.
    opts::detector detector_opts;
    opts::magnetic_field bfield_opts;
    opts::input_data input_opts;
    opts::clusterization clusterization_opts;
    opts::track_seeding seeding_opts;
    opts::track_finding finding_opts;
    opts::track_propagation propagation_opts;
    opts::track_fitting fitting_opts;
    opts::throughput throughput_opts;
    opts::logging logging_opts;
    opts::program_options program_opts{
        description,
        {detector_opts, bfield_opts, input_opts, clusterization_opts,
         seeding_opts, finding_opts, propagation_opts, fitting_opts,
         throughput_opts, logging_opts},
        argc,
        argv,
        prelogger->cloneWithSuffix("Options")};

    TRACCC_LOCAL_LOGGER(
        prelogger->clone(std::nullopt, traccc::Logging::Level(logging_opts)));

    char const * const pipeline = getenv("EFTRACKING_PIPELINE");
    if (pipeline == nullptr) {
        throw std::runtime_error("environment variable EFTRACKING_PIPELINE must be defined");
    }
    TRACCC_INFO("running algo for EFTracking pipeline " << pipeline);

    // Set up the timing info holder.
    performance::timing_info times;

    // Memory resource to use in the test.
    vecmem::host_memory_resource host_mr;

    // Construct the detector description object.
    traccc::detector_design_description::host det_descr{host_mr};
    traccc::detector_conditions_description::host det_cond{host_mr};
    traccc::io::read_detector_description(
        det_descr, det_cond, detector_opts.detector_file,
        detector_opts.digitization_file, detector_opts.conditions_file,
        traccc::data_format::json);

    // Construct a Detray detector object, if supported by the configuration.
    traccc::host_detector detector;
    traccc::io::read_detector(detector, host_mr, detector_opts.detector_file,
                              detector_opts.material_file,
                              detector_opts.grid_file);

    // Construct the magnetic field object.
    const auto field = details::make_magnetic_field(bfield_opts);

    // Read in all input events into memory.
    vecmem::vector<edm::silicon_cell_collection::host> input{&host_mr};
    {
        performance::timer t{"File reading", times};
        // Read the input cells into memory event-by-event.
        input.reserve(input_opts.events);
        for (std::size_t i = input_opts.skip;
             i < input_opts.skip + input_opts.events; ++i) {
            input.emplace_back(host_mr);
            static constexpr bool DEDUPLICATE = true;
            io::read_cells(input.back(), i, input_opts.directory,
                           logger().clone(), &det_cond, input_opts.format,
                           DEDUPLICATE, input_opts.use_acts_geom_source);
        }
    }

    // Algorithm configuration(s).
    detray::propagation::config propagation_config(propagation_opts);

    typename FULL_CHAIN_ALG::clustering_algorithm::config_type clustering_cfg(
        clusterization_opts);
    const traccc::seedfinder_config seedfinder_config(seeding_opts);
    const traccc::seedfilter_config seedfilter_config(seeding_opts);
    traccc::gbts_seedfinder_config gbts_config;

    const traccc::track_params_estimation_config track_params_estimation_config;

    typename FULL_CHAIN_ALG::finding_algorithm::config_type finding_cfg(
        finding_opts);
    finding_cfg.propagation = propagation_config;

    typename FULL_CHAIN_ALG::fitting_algorithm::config_type fitting_cfg(
        fitting_opts);
    fitting_cfg.propagation = propagation_config;

    seedfinder_config.zMin = -3000.f * unit<float>::mm;
    seedfinder_config.zMax = 3000.f * unit<float>::mm;
    seedfinder_config.rMax = 320.f * unit<float>::mm;
    seedfinder_config.rMin = 33.f * unit<float>::mm;
    // 3 sigmas of max beam spot delta z for Run 3
    // https://twiki.cern.ch/twiki/pub/AtlasPublic/BeamSpotPublicResults/BeamspotRun2vLumi_sig_z.png
    seedfinder_config.collisionRegionMax = 3 * 38 * unit<float>::mm;
    seedfinder_config.collisionRegionMin = -seedfinder_config.collisionRegionMax;
    // Based on Pixel barrel layers layout:
    // https://cds.cern.ch/record/2850865/files/ITK_schematic.png
    // Barrel layers position: 33, 96, 126, 225, 286
    // Minimum R distance between 2 layers: 30
    // Max distance between N, N+2 layer (allowing one "hole"): 160
    // Plus some margin (2 mm)
    seedfinder_config.deltaRMin = 20 * unit<float>::mm;
    seedfinder_config.deltaRMax = 100 * unit<float>::mm;
    seedfinder_config.deltaZMax = 800 * unit<float>::mm;

    seedfinder_config.minPt = 900.f * unit<float>::MeV;
    seedfinder_config.cotThetaMax = 27.2899f;
    seedfinder_config.impactMax = 2.f * unit<float>::mm;
    seedfinder_config.sigmaScattering = 3.0f;
    seedfinder_config.maxPtScattering = 10.f * unit<float>::GeV;
    seedfinder_config.radLengthPerSeed = 0.05f;
    seedfinder_config.maxSeedsPerSpM = 2;
    seedfinder_config.setup();

    const traccc::spacepoint_grid_config spacepoint_grid_config(
        seedfinder_config);

    seedfilter_config.good_spB_min_radius = 150.f * unit<float>::mm;
    seedfilter_config.good_spB_weight_increase = 400.f;
    seedfilter_config.good_spT_max_radius = 150.f * unit<float>::mm;
    seedfilter_config.good_spT_weight_increase = 200.f;
    seedfilter_config.good_spB_min_weight = 380.f;
    seedfilter_config.seed_min_weight = 200.f;
    seedfilter_config.spB_min_radius = 43.f * unit<float>::mm;
    seedfilter_config.compatSeedLimit = 1;

    finding_cfg.max_num_branches_per_seed = 3;
    finding_cfg.max_num_branches_per_surface = 1;
    finding_cfg.min_track_candidates_per_track = 7;
    finding_cfg.max_track_candidates_per_track = 20;
    finding_cfg.min_step_length_for_next_surface =
        0.5f * detray::unit<float>::mm;
    finding_cfg.max_step_counts_for_next_surface = 100;
    finding_cfg.chi2_max = 10.f;
    finding_cfg.max_num_skipping_per_cand = 2;

    finding_cfg.propagation.stepping.min_stepsize = 1e-4f * unit<float>::mm;
    finding_cfg.propagation.stepping.rk_error_tol = 1e-4f * unit<float>::mm;
    finding_cfg.propagation.stepping.step_constraint =
        std::numeric_limits<float>::max();
    finding_cfg.propagation.stepping.path_limit = 5.f * unit<float>::m;
    finding_cfg.propagation.stepping.max_rk_updates = 10000u;
    finding_cfg.propagation.stepping.use_mean_loss = true;
    finding_cfg.propagation.stepping.use_eloss_gradient = false;
    finding_cfg.propagation.stepping.use_field_gradient = false;
    finding_cfg.propagation.stepping.do_covariance_transport = true;
    finding_cfg.propagation.navigation.intersection.overstep_tolerance =
        -300.f * unit<float>::um;

    finding_cfg.max_num_tracks_per_measurement = 1;
    finding_cfg.initial_links_per_seed = 20;

    fitting_cfg.propagation.navigation.intersection.min_mask_tolerance =
        1e-5f * unit<float>::mm;
    fitting_cfg.propagation.navigation.intersection.max_mask_tolerance =
        3.f * unit<float>::mm;
    fitting_cfg.propagation.navigation.intersection.overstep_tolerance =
        -300.f * unit<float>::um;
    fitting_cfg.propagation.navigation.search_window[0] = 0u;
    fitting_cfg.propagation.navigation.search_window[1] = 0u;

    // Set up the full-chain algorithm.
    std::unique_ptr<FULL_CHAIN_ALG> alg = std::make_unique<FULL_CHAIN_ALG>(
        host_mr, clustering_cfg, seedfinder_config, spacepoint_grid_config,
        seedfilter_config, gbts_config, track_params_estimation_config,
        finding_cfg, fitting_cfg, det_descr, det_cond, field, &detector,
        logger().clone("FullChainAlg"));

    // Seed the random number generator.
    if (throughput_opts.random_seed == 0) {
        std::srand(static_cast<unsigned int>(std::time(0)));
    } else {
        std::srand(throughput_opts.random_seed);
    }

    // Set up a lambda that calls the correct function on the algorithm.
    std::function<std::size_t(const edm::silicon_cell_collection::host&)>
        process_event;
    if (throughput_opts.reco_stage == opts::throughput::stage::seeding) {
        process_event = [&](const edm::silicon_cell_collection::host& cells)
            -> std::size_t { return alg->seeding(cells).size(); };
    } else if (throughput_opts.reco_stage == opts::throughput::stage::full) {
        process_event = [&](const edm::silicon_cell_collection::host& cells)
            -> std::size_t { return (*alg)(cells).size(); };
    } else {
        throw std::invalid_argument("Unknown reconstruction stage");
    }

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip any step
    std::size_t rec_track_params = 0;

    // Cold Run events. To discard any "initialisation issues" in the
    // measurements.
    {
        // Set up a progress bar for the warm-up processing.
        indicators::ProgressBar progress_bar{
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{"Warm-up processing "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{throughput_opts.cold_run_events}};

        // Measure the time of execution.
        performance::timer t{"Warm-up processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_opts.cold_run_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                (throughput_opts.deterministic_event_order
                     ? i
                     : static_cast<std::size_t>(std::rand())) %
                input_opts.events;

            // Process one event.
            rec_track_params += process_event(input[event]);
            progress_bar.tick();
        }
    }

    // Reset the dummy counter.
    rec_track_params = 0;

    {
        TRACCC_INFO("start of event processing");

        // Set up a progress bar for the event processing.
        indicators::ProgressBar progress_bar{
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{"Event processing   "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{throughput_opts.processed_events}};

        // Measure the total time of execution.
        performance::timer t{"Event processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_opts.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                (throughput_opts.deterministic_event_order
                     ? i
                     : static_cast<std::size_t>(std::rand())) %
                input_opts.events;

            // Process one event.
            rec_track_params += process_event(input[event]);
            progress_bar.tick();
        }

        TRACCC_INFO("end of event processing");
    }

    // Explicitly delete the objects in the correct order.
    alg.reset();

    // Print some results.
    std::cout << "Reconstructed track parameters: " << rec_track_params
              << std::endl;
    std::cout << "Time totals:" << std::endl;
    std::cout << times << std::endl;
    std::cout << "Throughput:" << std::endl;
    std::cout << performance::throughput{throughput_opts.cold_run_events, times,
                                         "Warm-up processing"}
              << "\n"
              << performance::throughput{throughput_opts.processed_events,
                                         times, "Event processing"}
              << std::endl;

    // Return gracefully.
    return 0;
}

}  // namespace traccc
