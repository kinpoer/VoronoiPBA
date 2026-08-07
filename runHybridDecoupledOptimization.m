% Paper: Topology-aware impedance modelling and inverse design of Voronoi plate-lattices for bidirectional sound absorption
% Contact: Jin Bo, Email: kinpoer@nuaa.edu.cn
%
%RUNHYBRIDDECOUPLEDOPTIMIZATION Optimize one random Voronoi TpIM absorber.
%
% Keep this script and TpIMFunctions.m in the same folder. The default
% settings reproduce the four-stage spatial-frequency cascade used for the
% single-sided design, followed by acoustic interpretation and SCDM export.

clear;
clc;
close all;
tic;

scriptFolder = fileparts(mfilename("fullpath"));
addpath(scriptFolder);
outputFolder = fullfile(scriptFolder, "HybridDecoupledOptimizationResults");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% User configuration

config.geometry.cellCount = [2, 2, 2];
config.geometry.cellSizeMm = 25;
config.geometry.seedCountRange = [3, 5];
config.geometry.randomSeed = 202607;
config.geometry.enableCvt = true;
config.geometry.cvtIterations = 20;
config.geometry.vertexMergeToleranceMm = 0.1;
config.geometry.clipDecimalDigits = 6;
config.geometry.showInitialTopology = true;
config.geometry.showFinalTopology = true;

config.model.isBidirectional = false;
config.model.wallThicknessMm = 1.0;
config.model.topThicknessMm = 1.0;
config.model.incidentDomainHeightMm = 20.0;
config.model.holeAreaThresholdMm2 = 8.0;
config.model.fluidGeometryToleranceMm = 0.1;

config.design.topRadiusRangeMm = [0.35, 2.5];
config.design.topSpacingRangeMm = [1.2, 7.0];
config.design.internalRadiusRangeMm = [0.35, 2.5];
config.design.internalSpacingRangeMm = [1.2, 7.0];
config.design.clusteringMethods = ["spectral", "modal", "fuzzy"];
config.design.selectedClusteringMethod = "fuzzy";
config.design.renderClustering = true;

config.material.airDensityKgM3 = 1.21;
config.material.soundSpeedMS = 343;
config.material.dynamicViscosityPaS = 1.84e-5;

% Frequency definitions use [start frequency (Hz), stop frequency (Hz), samples].
config.frequency.low = [200, 500, 50];
config.frequency.mid = [400, 800, 50];
config.frequency.high = [700, 1600, 50];
config.frequency.global = [200, 1600, 150];
config.frequency.show = [20, 1600, 150];

config.ga.populationSize = 100;
config.ga.maxGenerations = 50;
config.ga.useParallel = true;
config.ga.randomSeed = 202607;
config.ga.stageWeights.high = [0.1, 0.2, 0.7];
config.ga.stageWeights.mid = [0.2, 0.7, 0.1];
config.ga.stageWeights.low = [0.8, 0.1, 0.1];

config.postprocess.visualizationFrequencyHz = 800;
config.postprocess.complexFrequencyRealRangeHz = config.frequency.show(1:2);
config.postprocess.complexFrequencyImaginaryRangeHz = [-600, 600];
config.postprocess.complexFrequencyRealSamples = 60;
config.postprocess.complexFrequencyImaginarySamples = 60;
config.output.folder = outputFolder;

config = TpIMFunctions.validateConfiguration(config, "cascade optimization");

%% Frequency data

disp("=> [1/7] Constructing frequency grids...");
frequencyData = TpIMFunctions.createFrequencyData(config);

%% Voronoi geometry

disp("=> [2/7] Generating and relaxing the random bounded Voronoi geometry...");
geometry = TpIMFunctions.buildVoronoiGeometry(config);

%% TpIM model

disp("=> [3/7] Constructing the fluid-cavity topology and TpIM network...");
model = TpIMFunctions.buildTpIMModel(geometry, config);

%% Design space

disp("=> [4/7] Constructing safe gene bounds and spatial memberships...");
designSpace = TpIMFunctions.createDesignSpace(geometry, model, config);
fprintf("   Optimizable faces: %d; genes: %d.\n", ...
    designSpace.numberOfOptimizableFaces, designSpace.numberOfGenes);

%% Cascade optimization

disp("=> [5/7] Running high-, middle-, low-frequency stages and global fine-tuning...");
results = TpIMFunctions.runCascadeOptimization( ...
    config, frequencyData, model, designSpace);
save(fullfile(outputFolder, "cascadeOptimizationResults.mat"), ...
    "config", "frequencyData", "geometry", "model", "designSpace", "results", "-v7.3");

%% Post-processing

disp("=> [6/7] Rendering stage responses and the final absorption/impedance curves...");
stageTitles = [ ...
    "Stage 1: High-frequency gateway", ...
    "Stage 2: Middle-frequency transition", ...
    "Stage 3: Low-frequency resonance", ...
    "Stage 4: Global impedance matching"];
stageColors = [ ...
    0.20, 0.50, 0.90;
    0.20, 0.75, 0.25;
    0.85, 0.25, 0.20;
    0.10, 0.10, 0.10];
stageBands = { ...
    frequencyData.bandRanges.high, ...
    frequencyData.bandRanges.mid, ...
    frequencyData.bandRanges.low, ...
    frequencyData.bandRanges.global};

for stageIndex = 1:4
    figure("Name", stageTitles(stageIndex), "Color", "w");
    if stageIndex < 4
        plotFrequencyHz = frequencyData.globalHz;
    else
        plotFrequencyHz = frequencyData.showHz;
    end
    plot(plotFrequencyHz, results.performanceHistory{stageIndex}, ...
        "LineWidth", 2.5, "Color", stageColors(stageIndex, :));
    hold on;
    highlightedBand = stageBands{stageIndex};
    patch([highlightedBand(1), highlightedBand(2), highlightedBand(2), highlightedBand(1)], ...
        [0, 0, 1.05, 1.05], stageColors(stageIndex, :), ...
        "FaceAlpha", 0.10, "EdgeColor", "none");
    title(stageTitles(stageIndex), "FontName", "Times New Roman", "FontWeight", "bold");
    xlabel("Frequency (Hz)");
    ylabel("Absorption coefficient");
    xlim(frequencyData.bandRanges.show);
    ylim([0, 1.05]);
    grid on;
    set(gca, "FontName", "Times New Roman", "FontSize", 11);
end

TpIMFunctions.renderAbsorptionAndImpedance( ...
    frequencyData.showHz, results.alphaShow, ...
    results.normalizedImpedance, frequencyData.bandRanges);

exportPolygons = model.polygons;
for faceIndex = 1:designSpace.numberOfOptimizableFaces
    edgeId = designSpace.optimizableEdgeIds(faceIndex);
    polygonId = model.edgeNodeIds(edgeId, 3);
    arrayCount = results.arrayCounts(edgeId);
    exportPolygons(polygonId).nSide = arrayCount;
    exportPolygons(polygonId).n1 = arrayCount;
    exportPolygons(polygonId).n2 = arrayCount;
    exportPolygons(polygonId).nHoles = arrayCount^2;
    exportPolygons(polygonId).rHole = results.apertureRadiiM(edgeId)*1e3;
    exportPolygons(polygonId).bSpace = results.apertureSpacingsM(edgeId)*1e3;
end

previousFolder = pwd;
folderCleanup = onCleanup(@() cd(previousFolder)); %#ok<NASGU>
cd(outputFolder);
TpIMFunctions.generateScdmScript( ...
    exportPolygons, model.cavities, ...
    config.model.wallThicknessMm, config.model.topThicknessMm, ...
    config.model.incidentDomainHeightMm, config.geometry.cellCount(1), ...
    config.geometry.cellCount(2), config.geometry.cellCount(3), ...
    config.geometry.cellSizeMm, config.model.isBidirectional);

%% Acoustic interpretation and export

disp("=> [7/7] Computing complex-frequency, full-band, and 3-D acoustic maps...");
complexConfig = struct();
complexConfig.fRealRange = config.postprocess.complexFrequencyRealRangeHz;
complexConfig.fImagRange = config.postprocess.complexFrequencyImaginaryRangeHz;
complexConfig.nReal = config.postprocess.complexFrequencyRealSamples;
complexConfig.nImag = config.postprocess.complexFrequencyImaginarySamples;

complexFrequencyMap = TpIMFunctions.renderComplexFrequencyPlane( ...
    results.arrayCounts, results.apertureRadiiM, results.apertureSpacingsM, ...
    complexConfig, model.numberOfNodes, model.finalVolumesM3, model.edgeNodeIds, ...
    model.geometricLengthsM, model.faceAreasM2, model.edgeTypes, ...
    model.wallThicknessM, model.topThicknessM, model.airDensityKgM3, ...
    model.soundSpeedMS, model.dynamicViscosityPaS, model.totalSurfaceAreaM2, ...
    model.topIncidenceEdgeIds);
save(fullfile(outputFolder, "complexFrequencyPlaneMap.mat"), ...
    "complexFrequencyMap", "complexConfig");

acousticState = TpIMFunctions.computeFullBandAcousticState( ...
    results.arrayCounts, results.apertureRadiiM, results.apertureSpacingsM, ...
    frequencyData.globalHz, frequencyData.globalAngularFrequency, ...
    config.postprocess.visualizationFrequencyHz, model.cavities, ...
    model.numberOfNodes, model.finalVolumesM3, model.edgeNodeIds, ...
    model.geometricLengthsM, model.faceAreasM2, model.edgeTypes, ...
    model.wallThicknessM, model.topThicknessM, model.airDensityKgM3, ...
    model.soundSpeedMS, model.dynamicViscosityPaS, model.isBidirectional, ...
    model.totalSurfaceAreaM2, model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds);
TpIMFunctions.renderFullBandPhysics(frequencyData.globalHz, acousticState, model.cavities);
TpIMFunctions.renderPolyhedralAcousticMaps( ...
    model.cavities, exportPolygons, geometry.final.vertices, model.finalNodesMm, ...
    model.edgeNodeIds, model.edgeTypes, results.arrayCounts, model.faceAreasMm2, ...
    acousticState, model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds, ...
    model.isBidirectional);

fprintf("=> Complete. Final mean absorption: %.6f. Results: %s\n", ...
    mean(results.alphaShow), outputFolder);
toc;
