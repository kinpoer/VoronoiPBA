% Paper: Topology-aware impedance modelling and inverse design of Voronoi plate-lattices for bidirectional sound absorption
% Contact: Jin Bo, Email: kinpoer@nuaa.edu.cn
%
%RUNM1M6SCALINGSTUDY Compare M1 and M6 as the Voronoi array grows.
%
% Each array size receives one reproducible random geometry. M1 and M6 are
% then run independently under the same TpIM-evaluation budget. 

clear;
clc;
close all;
tic;

scriptFolder = fileparts(mfilename("fullpath"));
addpath(scriptFolder);
outputFolder = fullfile(scriptFolder, "M1M6ScalingResults");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% User configuration

scalingConfig.cellCountCases = [ ...
    3, 3, 3;
    4, 4, 3;
    5, 5, 3];
scalingConfig.geometryBaseSeed = 202607;
scalingConfig.numberOfRuns = 10;
scalingConfig.efficiencyAlphaTargets = [0.45, 0.5, 0.55];
scalingConfig.primaryAlphaTarget = 0.5;
scalingConfig.outputDir = outputFolder;

config.geometry.cellCount = scalingConfig.cellCountCases(1, :);
config.geometry.cellSizeMm = 20;
config.geometry.seedCountRange = [3, 5];
config.geometry.randomSeed = scalingConfig.geometryBaseSeed;
config.geometry.enableCvt = true;
config.geometry.cvtIterations = 20;
config.geometry.vertexMergeToleranceMm = 0.1;
config.geometry.clipDecimalDigits = 6;
config.geometry.showInitialTopology = false;
config.geometry.showFinalTopology = false;

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
config.design.clusteringMethods = "fuzzy";
config.design.selectedClusteringMethod = "fuzzy";
config.design.renderClustering = false;

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

config.comparison.methodCodes = {"M1", "M6"};
config.comparison.methodNames = { ...
    "M1 Direct full-variable global GA", ...
    "M6 Complete method with global fine-tuning"};
config.comparison.nRuns = scalingConfig.numberOfRuns;
config.comparison.populationSize = config.ga.populationSize;
config.comparison.totalEvalBudget = ...
    config.ga.populationSize*(config.ga.maxGenerations + 1);
config.comparison.useParallel = config.ga.useParallel;
config.comparison.overlapThreshold = 0.25;
config.comparison.diagnosticGlobalObjective = true;
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
config.comparison.efficiencyAlphaTargets = ...
    scalingConfig.efficiencyAlphaTargets;
config.comparison.convergenceGridPoints = 300;

config = TpIMFunctions.validateConfiguration(config, "M1-M6 scaling study");
config.comparison.useParallel = config.ga.useParallel;
frequencyData = TpIMFunctions.createFrequencyData(config);

allScalingRows = table();
allCaseResults = cell(size(scalingConfig.cellCountCases, 1), 1);

%% Scale-case loop

for caseIndex = 1:size(scalingConfig.cellCountCases, 1)
    caseCellCount = scalingConfig.cellCountCases(caseIndex, :);
    caseLabel = sprintf("%dx%dx%d", caseCellCount);
    caseOutputFolder = fullfile(outputFolder, "case_" + caseLabel);
    if ~isfolder(caseOutputFolder)
        mkdir(caseOutputFolder);
    end

    fprintf("\n=> [Scaling] Case %s (%d/%d)\n", caseLabel, caseIndex, ...
        size(scalingConfig.cellCountCases, 1));
    caseConfig = config;
    caseConfig.geometry.cellCount = caseCellCount;
    caseConfig.geometry.randomSeed = scalingConfig.geometryBaseSeed + caseIndex;
    caseConfig.comparison.baseSeed = ...
        scalingConfig.geometryBaseSeed + 1000*caseIndex;
    caseConfig.comparison.outputDir = caseOutputFolder;

    geometry = TpIMFunctions.buildVoronoiGeometry(caseConfig);
    model = TpIMFunctions.buildTpIMModel(geometry, caseConfig);
    designSpace = TpIMFunctions.createDesignSpace(geometry, model, caseConfig);
    fprintf("   Optimizable faces: %d; genes: %d.\n", ...
        designSpace.numberOfOptimizableFaces, designSpace.numberOfGenes);

    caseResults = TpIMFunctions.runMethodComparison( ...
        caseConfig, frequencyData, model, designSpace);
    allCaseResults{caseIndex} = caseResults;
    [~, caseSummaryTable] = ...
        TpIMFunctions.renderMethodComparisonFigure( ...
        caseResults, caseConfig.comparison);
    save(fullfile(caseOutputFolder, "m1M6ScalingCaseResults.mat"), ...
        "caseResults", "caseConfig", "caseSummaryTable", "caseCellCount", ...
        "-v7.3");

    numberOfRows = numel(caseResults);
    caseLabels = repmat(string(caseLabel), numberOfRows, 1);
    nx = repmat(caseCellCount(1), numberOfRows, 1);
    ny = repmat(caseCellCount(2), numberOfRows, 1);
    nz = repmat(caseCellCount(3), numberOfRows, 1);
    numberOfCells = repmat(prod(caseCellCount), numberOfRows, 1);
    numberOfFaces = repmat( ...
        designSpace.numberOfOptimizableFaces, numberOfRows, 1);
    numberOfGenes = repmat(designSpace.numberOfGenes, numberOfRows, 1);
    methods = strings(numberOfRows, 1);
    runIds = zeros(numberOfRows, 1);
    finalObjectives = zeros(numberOfRows, 1);
    meanGlobalAbsorption = zeros(numberOfRows, 1);
    tpimEvaluations = zeros(numberOfRows, 1);
    targetEvaluations = nan(numberOfRows, ...
        numel(scalingConfig.efficiencyAlphaTargets));
    targetReached = false(numberOfRows, ...
        numel(scalingConfig.efficiencyAlphaTargets));

    for resultIndex = 1:numberOfRows
        result = caseResults(resultIndex);
        methods(resultIndex) = string(result.methodCode);
        runIds(resultIndex) = result.run;
        finalObjectives(resultIndex) = result.finalObjective;
        meanGlobalAbsorption(resultIndex) = result.meanAlphaGlobal;
        tpimEvaluations(resultIndex) = result.totalTpimEvals;
        bestAbsorptionTrace = -result.traceGlobalBestf(:);
        for targetIndex = 1:numel(scalingConfig.efficiencyAlphaTargets)
            targetEvaluations(resultIndex, targetIndex) = ...
                TpIMFunctions.findFirstEvaluationAtTarget( ...
                result.traceEval, bestAbsorptionTrace, ...
                scalingConfig.efficiencyAlphaTargets(targetIndex));
            targetReached(resultIndex, targetIndex) = ...
                isfinite(targetEvaluations(resultIndex, targetIndex));
        end
    end

    caseTable = table(caseLabels, nx, ny, nz, numberOfCells, ...
        numberOfFaces, numberOfGenes, methods, runIds, finalObjectives, ...
        meanGlobalAbsorption, tpimEvaluations, ...
        'VariableNames', {'CaseLabel', 'Nx', 'Ny', 'Nz', 'NumCells', ...
        'W', 'Nvars', 'Method', 'Run', 'FinalObjective', ...
        'MeanAlphaGlobal', 'TpIMEvaluations'});
    for targetIndex = 1:numel(scalingConfig.efficiencyAlphaTargets)
        suffix = sprintf("%d", targetIndex);
        caseTable.("AlphaTarget" + suffix) = repmat( ...
            scalingConfig.efficiencyAlphaTargets(targetIndex), numberOfRows, 1);
        caseTable.("EvalToTarget" + suffix) = ...
            targetEvaluations(:, targetIndex);
        caseTable.("ReachedTarget" + suffix) = ...
            targetReached(:, targetIndex);
    end

    if isempty(allScalingRows)
        allScalingRows = caseTable;
    else
        allScalingRows = [allScalingRows; caseTable]; 
    end
    writetable(allScalingRows, ...
        fullfile(outputFolder, "m1M6ScalingRawRuns.csv"));
    save(fullfile(outputFolder, "m1M6ScalingPartial.mat"), ...
        "allScalingRows", "allCaseResults", "scalingConfig", "-v7.3");
end

%% Aggregate scale-effect figure

[~, scalingSummaryTable, scalingSpeedupTable] = ...
    TpIMFunctions.renderScalingStudyFigure(allScalingRows, scalingConfig);
save(fullfile(outputFolder, "m1M6ScalingResults.mat"), ...
    "allScalingRows", "allCaseResults", "scalingConfig", ...
    "scalingSummaryTable", "scalingSpeedupTable", "-v7.3");

fprintf("=> Complete. Scaling results: %s\n", outputFolder);
toc;
