% Paper: Topology-aware impedance modelling and inverse design of Voronoi plate-lattices for bidirectional sound absorption
% Contact: Jin Bo, Email: kinpoer@nuaa.edu.cn
%
%RUNM1M6ABLATIONSTUDY Compare six optimization variants under a matched TpIM budget.
%
% The script generates one random Voronoi structure, reuses the common TpIM
% workflow, runs M1-M6 independently, and exports ablation figures and summary data.

clear;
clc;
close all;
tic;

scriptFolder = fileparts(mfilename("fullpath"));
addpath(scriptFolder);
outputFolder = fullfile(scriptFolder, "M1M6AblationResults");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% User configuration

config.geometry.cellCount = [4, 4, 3];
config.geometry.cellSizeMm = 20;
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

config.frequency.low = [20,  220,  20];
config.frequency.mid = [180,  420,  20];
config.frequency.high = [380,  600,  20];
config.frequency.global = [20,  600,  60];
config.frequency.show = [20, 600, 150];

config.ga.populationSize = 400;
config.ga.maxGenerations = 280;
config.ga.useParallel = true;
config.ga.stageWeights.high = [0.1, 0.2, 0.7];
config.ga.stageWeights.mid = [0.2, 0.7, 0.1];
config.ga.stageWeights.low = [0.8, 0.1, 0.1];

config.comparison.methodCodes = {"M1", "M2", "M3", "M4", "M5", "M6"};
config.comparison.methodNames = { ...
    "M1 Direct full-variable global GA", ...
    "M2 Frequency-staged GA without spatial partition", ...
    "M3 Hard-cluster partition with cascade GA", ...
    "M4 Fuzzy-overlap partition without top-down cascade", ...
    "M5 Full spatial-frequency cascade without global fine-tuning", ...
    "M6 Complete method with global fine-tuning"};
config.comparison.nRuns = 20;
config.comparison.populationSize = config.ga.populationSize;
config.comparison.totalEvalBudget = config.ga.populationSize*(config.ga.maxGenerations + 1);
config.comparison.useParallel = config.ga.useParallel;
config.comparison.baseSeed = 202607;
config.comparison.overlapThreshold = 0.25;
config.comparison.diagnosticGlobalObjective = true;
config.comparison.outputDir = outputFolder;
config.comparison.stageWeights = config.ga.stageWeights;
config.comparison.stageBudget.mode = "weighted";
config.comparison.stageBudget.minGenerations = 1;
config.comparison.stageBudget.variableExponent = 0.65;
config.comparison.stageBudget.bandwidthExponent = 0.35;
config.comparison.stageBudget.typeWeights.high = 1.10;
config.comparison.stageBudget.typeWeights.mid = 1.00;
config.comparison.stageBudget.typeWeights.low = 1.15;
config.comparison.stageBudget.typeWeights.fine = 1.40;
config.comparison.stageBudget.typeWeights.global = 1.00;
config.comparison.stageSwitch.enabled = true;
config.comparison.stageSwitch.stallGenerations = ...
    min(12, max(4, round(0.08*config.ga.maxGenerations)));
config.comparison.stageSwitch.functionTolerance = 1e-5;
config.comparison.convergencePlot.smoothWindowFraction = 0.04;
config.comparison.convergencePlot.markFailedTargets = true;
config.comparison.efficiencyAlphaTargets = [0.45, 0.50, 0.55];
config.comparison.convergenceGridPoints = 300;

config = TpIMFunctions.validateConfiguration(config, "M1-M6 ablation study");
config.comparison.useParallel = config.ga.useParallel;

%% Frequency data

frequencyData = TpIMFunctions.createFrequencyData(config);

%% Voronoi geometry

disp("=> [1/4] Generating the shared random Voronoi geometry...");
geometry = TpIMFunctions.buildVoronoiGeometry(config);

%% TpIM model

disp("=> [2/4] Constructing the shared TpIM model...");
model = TpIMFunctions.buildTpIMModel(geometry, config);

%% Design space

disp("=> [3/4] Constructing safe genes and spatial memberships...");
designSpace = TpIMFunctions.createDesignSpace(geometry, model, config);
fprintf("   Optimizable faces: %d; genes: %d.\n", ...
    designSpace.numberOfOptimizableFaces, designSpace.numberOfGenes);

%% M1-M6 comparison

disp("=> [4/4] Running the matched-budget M1-M6 comparison...");
comparisonResults = TpIMFunctions.runMethodComparison( ...
    config, frequencyData, model, designSpace);
[~, summaryTable] = TpIMFunctions.renderMethodComparisonFigure( ...
    comparisonResults, config.comparison);

save(fullfile(outputFolder, "fig13M1M6ComparisonResults.mat"), ...
    "comparisonResults", "config", "frequencyData", "summaryTable", "-v7.3");

methodCodes = string({comparisonResults.methodCode});
m6ResultIds = find(methodCodes == "M6");
[~, bestLocalId] = min([comparisonResults(m6ResultIds).finalObjective]);
bestM6 = comparisonResults(m6ResultIds(bestLocalId));

[arrayCounts, apertureRadiiM, apertureSpacingsM] = TpIMFunctions.decodeGenes( ...
    bestM6.x, designSpace.numberOfOptimizableFaces, model.numberOfEdges, ...
    designSpace.optimizableEdgeIds, model.safeArrayLengthsMm);
[alphaShow, ~, ~, ~, normalizedImpedance] = TpIMFunctions.solveAcoustics( ...
    arrayCounts, apertureRadiiM, apertureSpacingsM, frequencyData.showHz, ...
    frequencyData.showAngularFrequency, model.numberOfNodes, model.finalVolumesM3, ...
    model.edgeNodeIds, model.geometricLengthsM, model.faceAreasM2, model.edgeTypes, ...
    model.wallThicknessM, model.topThicknessM, model.airDensityKgM3, ...
    model.soundSpeedMS, model.dynamicViscosityPaS, model.isBidirectional, ...
    model.totalSurfaceAreaM2, model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds);
TpIMFunctions.renderAbsorptionAndImpedance( ...
    frequencyData.showHz, alphaShow, normalizedImpedance, frequencyData.bandRanges);

exportPolygons = model.polygons;
for faceIndex = 1:designSpace.numberOfOptimizableFaces
    edgeId = designSpace.optimizableEdgeIds(faceIndex);
    polygonId = model.edgeNodeIds(edgeId, 3);
    sideCount = arrayCounts(edgeId);
    exportPolygons(polygonId).nSide = sideCount;
    exportPolygons(polygonId).n1 = sideCount;
    exportPolygons(polygonId).n2 = sideCount;
    exportPolygons(polygonId).nHoles = sideCount^2;
    exportPolygons(polygonId).rHole = apertureRadiiM(edgeId)*1e3;
    exportPolygons(polygonId).bSpace = apertureSpacingsM(edgeId)*1e3;
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

fprintf("=> Complete. Best M6 run: %d; objective: %.6f; mean absorption: %.6f.\n", ...
    bestM6.run, bestM6.finalObjective, bestM6.meanAlphaGlobal);
fprintf("   Results: %s\n", outputFolder);
toc;
