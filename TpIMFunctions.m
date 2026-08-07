% Paper: Topology-aware impedance modelling and inverse design of Voronoi plate-lattices for bidirectional sound absorption
% Contact: Jin Bo, Email: kinpoer@nuaa.edu.cn
%
classdef TpIMFunctions
%TPIMFUNCTIONS Reusable geometry, acoustic, optimization, and visualization utilities.
%
% Requires MATLAB R2021b or later. Core workflows use Optimization
% Toolbox, Global Optimization Toolbox, Statistics and Machine Learning
% Toolbox, Fuzzy Logic Toolbox, and Image Processing Toolbox. Parallel
% Computing Toolbox is optional.

methods (Static)

function config = validateConfiguration(config, workflowName)
%VALIDATECONFIGURATION Validate shared geometry, model, frequency, and GA settings.

    if nargin < 2
        workflowName = "workflow";
    end
    if verLessThan("matlab", "9.11")
        error("TpIMFunctions:UnsupportedMatlab", ...
            "The %s requires MATLAB R2021b or later.", workflowName);
    end

    requiredProducts = [ ...
        "Optimization Toolbox", ...
        "Global Optimization Toolbox", ...
        "Statistics and Machine Learning Toolbox", ...
        "Fuzzy Logic Toolbox", ...
        "Image Processing Toolbox"];
    installedProductInfo = ver;
    installedProducts = string({installedProductInfo.Name});
    missingProducts = setdiff(requiredProducts, installedProducts);
    if ~isempty(missingProducts)
        error("TpIMFunctions:MissingProducts", ...
            "Missing required MathWorks products: %s.", strjoin(missingProducts, ", "));
    end

    validateattributes(config.geometry.cellCount, {'numeric'}, ...
        {'row', 'numel', 3, 'integer', 'positive'});
    validateattributes(config.geometry.cellSizeMm, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    validateattributes(config.geometry.seedCountRange, {'numeric'}, ...
        {'row', 'numel', 2, 'integer', 'nonnegative'});
    if config.geometry.seedCountRange(1) > config.geometry.seedCountRange(2)
        error("TpIMFunctions:InvalidSeedRange", ...
            "geometry.seedCountRange must be ordered as [minimum, maximum].");
    end
    if config.geometry.seedCountRange(2) < 1
        error("TpIMFunctions:EmptySeedRange", ...
            "geometry.seedCountRange must permit at least one seed per cell.");
    end
    validateattributes(config.geometry.randomSeed, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'});
    validateattributes(config.geometry.cvtIterations, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative'});
    validateattributes(config.ga.populationSize, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    validateattributes(config.ga.maxGenerations, {'numeric'}, ...
        {'scalar', 'integer', 'positive'});
    if isfield(config.ga, "randomSeed")
        validateattributes(config.ga.randomSeed, {'numeric'}, ...
            {'scalar', 'integer', 'nonnegative', 'finite'});
    end

    config.ga.useParallel = TpIMFunctions.resolveParallelSetting(config.ga.useParallel);
end

function frequencyData = createFrequencyData(config)
%CREATEFREQUENCYDATA Construct frequency vectors, angular frequencies, and band limits.

    bandNames = ["low", "mid", "high", "global", "show"];
    frequencyData = struct();
    frequencyData.bandRanges = struct();
    for bandName = bandNames
        definition = config.frequency.(bandName);
        validateattributes(definition, {'numeric'}, ...
            {'row', 'numel', 3, 'real', 'finite'});
        if definition(1) >= definition(2)
            error("TpIMFunctions:InvalidFrequencyRange", ...
                "frequency.%s must satisfy start frequency < stop frequency.", bandName);
        end
        if definition(3) < 2 || definition(3) ~= round(definition(3))
            error("TpIMFunctions:InvalidFrequencyCount", ...
                "frequency.%s sample count must be an integer of at least 2.", bandName);
        end
        frequencyHz = linspace(definition(1), definition(2), definition(3));
        frequencyData.(bandName + "Hz") = frequencyHz;
        frequencyData.(bandName + "AngularFrequency") = 2*pi*frequencyHz;
        frequencyData.bandRanges.(bandName) = definition(1:2);
    end
end

function useParallel = resolveParallelSetting(requestedUseParallel)
%RESOLVEPARALLELSETTING Use serial execution when Parallel Computing Toolbox is unavailable.

    useParallel = logical(requestedUseParallel);
    if useParallel && ~license('test', 'Distrib_Computing_Toolbox')
        warning("TpIMFunctions:ParallelUnavailable", ...
            "Parallel Computing Toolbox is unavailable. GA will run serially.");
        useParallel = false;
    end
end

function geometry = buildVoronoiGeometry(config)
%BUILDVORONOIGEOMETRY Generate random seeds, optional CVT relaxation, and bounded topology.

    cellCount = config.geometry.cellCount;
    cellSizeMm = config.geometry.cellSizeMm;
    extentMm = cellCount*cellSizeMm;
    cuboidVertices = [ ...
        0, 0, 0;
        extentMm(1), 0, 0;
        extentMm(1), extentMm(2), 0;
        0, extentMm(2), 0;
        0, 0, extentMm(3);
        extentMm(1), 0, extentMm(3);
        extentMm(1), extentMm(2), extentMm(3);
        0, extentMm(2), extentMm(3)];

    randomStream = RandStream("mt19937ar", "Seed", config.geometry.randomSeed);
    initialSeeds = TpIMFunctions.generateRandomSeeds( ...
        cellCount, cellSizeMm, config.geometry.seedCountRange, randomStream);
    initialState = TpIMFunctions.createGeometryState( ...
        initialSeeds, cuboidVertices, extentMm(3), config);

    relaxedSeeds = initialSeeds;
    if config.geometry.enableCvt
        for iteration = 1:config.geometry.cvtIterations
            [cvtVertices, cvtCells] = TpIMFunctions.buildBoundedVoronoi( ...
                relaxedSeeds, cuboidVertices, config.geometry.clipDecimalDigits);
            updatedSeeds = zeros(size(relaxedSeeds));
            for seedIndex = 1:numel(cvtCells)
                cellVertexIds = cvtCells{seedIndex};
                if isempty(cellVertexIds)
                    updatedSeeds(seedIndex, :) = relaxedSeeds(seedIndex, :);
                else
                    updatedSeeds(seedIndex, :) = mean(cvtVertices(cellVertexIds, :), 1);
                end
            end
            relaxedSeeds = updatedSeeds;
        end
    end
    finalState = TpIMFunctions.createGeometryState( ...
        relaxedSeeds, cuboidVertices, extentMm(3), config);

    geometry = struct();
    geometry.cellCount = cellCount;
    geometry.cellSizeMm = cellSizeMm;
    geometry.cuboidVertices = cuboidVertices;
    geometry.initial = initialState;
    geometry.final = finalState;
    geometry.cvtEnabled = logical(config.geometry.enableCvt);
    geometry.cvtIterations = config.geometry.cvtIterations;

    if config.geometry.showInitialTopology
        TpIMFunctions.renderTopologyExplorer( ...
            initialState.polygons, initialState.cavities, initialState.vertices);
    end
    if config.geometry.showFinalTopology
        TpIMFunctions.renderTopologyExplorer( ...
            finalState.polygons, finalState.cavities, finalState.vertices);
    end
end

function state = createGeometryState(seeds, cuboidVertices, maximumZMm, config)
%CREATEGEOMETRYSTATE Build one deduplicated bounded-Voronoi topology state.

    seeds = uniquetol(seeds, 1e-5, "ByRows", true, "DataScale", 1);
    [vertices, cells, cellStatus] = TpIMFunctions.buildBoundedVoronoi( ...
        seeds, cuboidVertices, config.geometry.clipDecimalDigits);
    [vertices, ~, vertexMap] = uniquetol(vertices, ...
        config.geometry.vertexMergeToleranceMm, "ByRows", true, "DataScale", 1);
    for cellIndex = 1:numel(cells)
        if ~isempty(cells{cellIndex})
            cells{cellIndex} = unique(vertexMap(cells{cellIndex}), "stable");
        end
    end
    [polygons, cavities, vertices] = TpIMFunctions.buildTopologyStructures( ...
        vertices, cells, maximumZMm, config.model.isBidirectional);

    state = struct();
    state.seeds = seeds;
    state.vertices = vertices;
    state.cells = cells;
    state.cellStatus = cellStatus;
    state.polygons = polygons;
    state.cavities = cavities;
end

function model = buildTpIMModel(geometry, config)
%BUILDTPIMMODEL Construct fluid cavities and the TpIM network in SI units.

    polygons = geometry.final.polygons;
    cavities = geometry.final.cavities;
    vertices = geometry.final.vertices;
    for cavityIndex = 1:numel(cavities)
        [fluidVertices, fluidFaces, realVolumeMm3, validFaceAreas] = ...
            TpIMFunctions.computeFluidCavityHalfspace( ...
            cavities(cavityIndex).PolygonIDs, polygons, ...
            config.model.wallThicknessMm, cavities(cavityIndex).NodeCoord, ...
            config.model.fluidGeometryToleranceMm);
        cavities(cavityIndex).FluidVerts = fluidVertices;
        cavities(cavityIndex).FluidFaces = fluidFaces;
        cavities(cavityIndex).RealVolume = realVolumeMm3;
        cavities(cavityIndex).ValidFacesAreaMap = validFaceAreas;
    end

    cellCount = geometry.cellCount;
    cellSizeMm = geometry.cellSizeMm;
    minimumSpacingMm = min([ ...
        config.design.topSpacingRangeMm(1), ...
        config.design.internalSpacingRangeMm(1)]);
    [finalNodesMm, numberOfNodes, finalVolumesMm3, edgeNodeIds, ...
        geometricLengthsMm, faceAreasMm2, edgeTypes, topIncidenceEdgeIds, ...
        bottomIncidenceEdgeIds, polygons, maximumArrayCounts, safeArrayLengthsMm] = ...
        TpIMFunctions.extractTpIMMatrixBounds( ...
        cavities, polygons, vertices, cellCount(1), cellCount(2), cellCount(3), ...
        cellSizeMm, 0.5*cellSizeMm, config.model.holeAreaThresholdMm2, ...
        config.model.wallThicknessMm, minimumSpacingMm, config.model.isBidirectional);

    model = struct();
    model.cavities = cavities;
    model.polygons = polygons;
    model.verticesMm = vertices;
    model.finalNodesMm = finalNodesMm;
    model.numberOfNodes = numberOfNodes;
    model.finalVolumesMm3 = finalVolumesMm3;
    model.finalVolumesM3 = finalVolumesMm3*1e-9;
    model.totalFluidVolumeMm3 = sum(finalVolumesMm3);
    model.edgeNodeIds = edgeNodeIds;
    model.numberOfEdges = size(edgeNodeIds, 1);
    model.geometricLengthsMm = geometricLengthsMm;
    model.geometricLengthsM = geometricLengthsMm*1e-3;
    model.faceAreasMm2 = faceAreasMm2;
    model.faceAreasM2 = faceAreasMm2*1e-6;
    model.edgeTypes = edgeTypes;
    model.topIncidenceEdgeIds = topIncidenceEdgeIds;
    model.bottomIncidenceEdgeIds = bottomIncidenceEdgeIds;
    model.maximumArrayCounts = maximumArrayCounts;
    model.safeArrayLengthsMm = safeArrayLengthsMm;
    model.wallThicknessM = config.model.wallThicknessMm*1e-3;
    model.topThicknessM = config.model.topThicknessMm*1e-3;
    model.totalSurfaceAreaM2 = prod(cellCount(1:2)*cellSizeMm)*1e-6;
    model.airDensityKgM3 = config.material.airDensityKgM3;
    model.soundSpeedMS = config.material.soundSpeedMS;
    model.dynamicViscosityPaS = config.material.dynamicViscosityPaS;
    model.isBidirectional = config.model.isBidirectional;
end

function designSpace = createDesignSpace(geometry, model, config)
%CREATEDESIGNSPACE Build safe gene bounds, initial genes, and spatial memberships.

    if isempty(geometry.final.seeds)
        error("TpIMFunctions:EmptyGeometry", "The final geometry contains no seeds.");
    end
    optimizableEdgeIds = find(model.maximumArrayCounts >= 1);
    numberOfFaces = numel(optimizableEdgeIds);
    if numberOfFaces == 0
        error("TpIMFunctions:NoOptimizableFaces", ...
            "No interface can contain a safe microperforation array under the current bounds.");
    end

    numberOfGenes = 3*numberOfFaces;
    lowerBounds = zeros(1, numberOfGenes);
    upperBounds = zeros(1, numberOfGenes);
    initialGenes = zeros(1, numberOfGenes);
    for faceIndex = 1:numberOfFaces
        edgeId = optimizableEdgeIds(faceIndex);
        if model.edgeTypes(edgeId) == 3 || model.edgeTypes(edgeId) == 4
            radiusRangeMm = config.design.topRadiusRangeMm;
            spacingRangeMm = config.design.topSpacingRangeMm;
        else
            radiusRangeMm = config.design.internalRadiusRangeMm;
            spacingRangeMm = config.design.internalSpacingRangeMm;
        end
        lowerBounds(faceIndex) = radiusRangeMm(1);
        upperBounds(faceIndex) = radiusRangeMm(2);
        lowerBounds(numberOfFaces + faceIndex) = spacingRangeMm(1);
        upperBounds(numberOfFaces + faceIndex) = spacingRangeMm(2);

        maximumCount = floor(model.safeArrayLengthsMm(edgeId)/spacingRangeMm(1));
        lowerBounds(2*numberOfFaces + faceIndex) = 0;
        upperBounds(2*numberOfFaces + faceIndex) = maximumCount;
        initialGenes(faceIndex) = mean(radiusRangeMm);
        initialGenes(numberOfFaces + faceIndex) = mean(spacingRangeMm);
        if maximumCount >= 1
            initialGenes(2*numberOfFaces + faceIndex) = max(1, round(maximumCount/2));
        end
    end

    clusteringMethods = string(config.design.clusteringMethods);
    clusteringResults = struct();
    for methodName = clusteringMethods
        [cavityGroups, faceGroups, membership] = TpIMFunctions.classifyCavities( ...
            char(methodName), model.cavities, model.edgeNodeIds, model.finalNodesMm, ...
            optimizableEdgeIds, numberOfFaces, config.design.renderClustering);
        fieldName = matlab.lang.makeValidName(methodName);
        clusteringResults.(fieldName) = struct( ...
            "cavityGroups", cavityGroups, ...
            "faceGroups", faceGroups, ...
            "membership", membership);
        if config.design.renderClustering
            figure("Name", methodName + " clustering", "Color", "w");
            TpIMFunctions.renderClusteredCavities( ...
                model.cavities, model.finalNodesMm, model.edgeNodeIds, ...
                optimizableEdgeIds, cavityGroups, faceGroups, 1.0);
            title(methodName + " clustering", "Interpreter", "none");
        end
    end

    selectedField = matlab.lang.makeValidName(config.design.selectedClusteringMethod);
    if ~isfield(clusteringResults, selectedField)
        error("TpIMFunctions:MissingSelectedClustering", ...
            "The selected clustering method was not evaluated: %s.", ...
            config.design.selectedClusteringMethod);
    end
    selected = clusteringResults.(selectedField);

    designSpace = struct();
    designSpace.optimizableEdgeIds = optimizableEdgeIds;
    designSpace.numberOfOptimizableFaces = numberOfFaces;
    designSpace.numberOfGenes = numberOfGenes;
    designSpace.lowerBounds = lowerBounds;
    designSpace.upperBounds = upperBounds;
    designSpace.initialGenes = initialGenes;
    designSpace.integerGeneIds = (2*numberOfFaces + 1):numberOfGenes;
    designSpace.clustering = clusteringResults;
    designSpace.selectedClusteringMethod = string(config.design.selectedClusteringMethod);
    designSpace.faceGroups = selected.faceGroups;
    designSpace.cavityMembership = selected.membership;
end

function results = runCascadeOptimization(config, frequencyData, model, designSpace)
%RUNCASCADEOPTIMIZATION Run high-, middle-, low-frequency stages and global fine-tuning.

    if isfield(config.ga, 'randomSeed')
        rng(config.ga.randomSeed, 'twister');
    end
    numberOfFaces = designSpace.numberOfOptimizableFaces;
    genes = designSpace.initialGenes;
    lowerBounds = designSpace.lowerBounds;
    upperBounds = designSpace.upperBounds;
    useParallel = TpIMFunctions.resolveParallelSetting(config.ga.useParallel);
    gaOptions = optimoptions("ga", ...
        "PopulationSize", config.ga.populationSize, ...
        "MaxGenerations", config.ga.maxGenerations, ...
        "Display", "off", ...
        "UseParallel", useParallel);

    stageDefinitions = struct( ...
        "name", {"high", "mid", "low"}, ...
        "groupId", {3, 2, 1}, ...
        "weights", {config.ga.stageWeights.high, ...
            config.ga.stageWeights.mid, config.ga.stageWeights.low});
    performanceHistory = cell(4, 1);
    stageLogs = repmat(struct("name", "", "activeFaceIds", [], "loss", NaN), 3, 1);

    for stageIndex = 1:numel(stageDefinitions)
        stage = stageDefinitions(stageIndex);
        activeFaceIds = find(designSpace.faceGroups == stage.groupId)';
        activeGeneIds = [activeFaceIds, numberOfFaces + activeFaceIds, ...
            2*numberOfFaces + activeFaceIds];
        stageLoss = NaN;
        if ~isempty(activeGeneIds)
            integerIds = (2*numel(activeFaceIds) + 1):numel(activeGeneIds);
            objective = @(subGenes) TpIMFunctions.calculateWeightedFitness( ...
                subGenes, activeGeneIds, genes, numberOfFaces, ...
                frequencyData.globalHz, frequencyData.globalAngularFrequency, ...
                frequencyData.bandRanges, stage.weights(1), stage.weights(2), ...
                stage.weights(3), model.numberOfNodes, model.finalVolumesM3, ...
                model.edgeNodeIds, model.geometricLengthsM, model.faceAreasM2, ...
                model.edgeTypes, model.wallThicknessM, model.topThicknessM, ...
                model.airDensityKgM3, model.soundSpeedMS, model.dynamicViscosityPaS, ...
                model.isBidirectional, model.totalSurfaceAreaM2, ...
                model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds, ...
                designSpace.optimizableEdgeIds, model.safeArrayLengthsMm);
            [optimizedGenes, stageLoss] = ga( ...
                objective, numel(activeGeneIds), [], [], [], [], ...
                lowerBounds(activeGeneIds), upperBounds(activeGeneIds), ...
                [], integerIds, gaOptions);
            genes(activeGeneIds) = optimizedGenes;
        end
        [arrayCounts, radiiMm, spacingsMm] = TpIMFunctions.decodeGenes( ...
            genes, numberOfFaces, model.numberOfEdges, ...
            designSpace.optimizableEdgeIds, model.safeArrayLengthsMm);
        performanceHistory{stageIndex} = TpIMFunctions.solveAcoustics( ...
            arrayCounts, radiiMm, spacingsMm, frequencyData.globalHz, ...
            frequencyData.globalAngularFrequency, model.numberOfNodes, ...
            model.finalVolumesM3, model.edgeNodeIds, model.geometricLengthsM, ...
            model.faceAreasM2, model.edgeTypes, model.wallThicknessM, ...
            model.topThicknessM, model.airDensityKgM3, model.soundSpeedMS, ...
            model.dynamicViscosityPaS, model.isBidirectional, ...
            model.totalSurfaceAreaM2, model.topIncidenceEdgeIds, ...
            model.bottomIncidenceEdgeIds);
        stageLogs(stageIndex).name = stage.name;
        stageLogs(stageIndex).activeFaceIds = activeFaceIds;
        stageLogs(stageIndex).loss = stageLoss;
    end

    fineLowerBounds = max(lowerBounds, 0.5*genes);
    fineUpperBounds = min(upperBounds, 1.5*genes);
    fineLowerBounds(designSpace.integerGeneIds) = ...
        floor(fineLowerBounds(designSpace.integerGeneIds));
    fineUpperBounds(designSpace.integerGeneIds) = ...
        ceil(fineUpperBounds(designSpace.integerGeneIds));
    globalObjective = @(candidateGenes) TpIMFunctions.calculateGlobalFitness( ...
        candidateGenes, numberOfFaces, frequencyData.globalHz, ...
        frequencyData.globalAngularFrequency, model.numberOfNodes, ...
        model.finalVolumesM3, model.edgeNodeIds, model.geometricLengthsM, ...
        model.faceAreasM2, model.edgeTypes, model.wallThicknessM, ...
        model.topThicknessM, model.airDensityKgM3, model.soundSpeedMS, ...
        model.dynamicViscosityPaS, model.isBidirectional, model.totalSurfaceAreaM2, ...
        model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds, ...
        designSpace.optimizableEdgeIds, model.safeArrayLengthsMm);
    fineOptions = optimoptions(gaOptions, ...
        "InitialPopulationMatrix", genes, ...
        "Display", "iter");
    [genes, finalLoss] = ga(globalObjective, designSpace.numberOfGenes, ...
        [], [], [], [], fineLowerBounds, fineUpperBounds, [], ...
        designSpace.integerGeneIds, fineOptions);

    [arrayCounts, radiiMm, spacingsMm] = TpIMFunctions.decodeGenes( ...
        genes, numberOfFaces, model.numberOfEdges, ...
        designSpace.optimizableEdgeIds, model.safeArrayLengthsMm);
    [alphaShow, ~, ~, ~, normalizedImpedance] = TpIMFunctions.solveAcoustics( ...
        arrayCounts, radiiMm, spacingsMm, frequencyData.showHz, ...
        frequencyData.showAngularFrequency, model.numberOfNodes, ...
        model.finalVolumesM3, model.edgeNodeIds, model.geometricLengthsM, ...
        model.faceAreasM2, model.edgeTypes, model.wallThicknessM, ...
        model.topThicknessM, model.airDensityKgM3, model.soundSpeedMS, ...
        model.dynamicViscosityPaS, model.isBidirectional, model.totalSurfaceAreaM2, ...
        model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds);
    performanceHistory{4} = alphaShow;

    results = struct();
    results.finalGenes = genes;
    results.finalLoss = finalLoss;
    results.arrayCounts = arrayCounts;
    results.apertureRadiiM = radiiMm;
    results.apertureSpacingsM = spacingsMm;
    results.performanceHistory = performanceHistory;
    results.stageLogs = stageLogs;
    results.alphaShow = alphaShow;
    results.normalizedImpedance = normalizedImpedance;
end

function results = runMethodComparison(config, frequencyData, model, designSpace)
%RUNMETHODCOMPARISON Run configured M1-M6 methods under a matched TpIM budget.

    comparisonConfig = config.comparison;
    comparisonConfig.methodCodes = cellstr(string(comparisonConfig.methodCodes));
    comparisonConfig.methodNames = cellstr(string(comparisonConfig.methodNames));
    comparisonConfig.bandRanges = frequencyData.bandRanges;
    comparisonConfig.fGlobal = frequencyData.globalHz;
    comparisonConfig.omegaGlobal = frequencyData.globalAngularFrequency;
    comparisonConfig.fShow = frequencyData.showHz;
    comparisonConfig.omegaShow = frequencyData.showAngularFrequency;
    results = TpIMFunctions.runMethodComparisonCore( ...
        comparisonConfig, designSpace.initialGenes, designSpace.lowerBounds, ...
        designSpace.upperBounds, designSpace.numberOfOptimizableFaces, ...
        model.numberOfEdges, designSpace.faceGroups, designSpace.cavityMembership, ...
        model.numberOfNodes, model.finalVolumesM3, ...
        model.edgeNodeIds, model.geometricLengthsM, model.faceAreasM2, ...
        model.edgeTypes, model.wallThicknessM, model.topThicknessM, ...
        model.airDensityKgM3, model.soundSpeedMS, model.dynamicViscosityPaS, ...
        model.isBidirectional, model.totalSurfaceAreaM2, ...
        model.topIncidenceEdgeIds, model.bottomIncidenceEdgeIds, ...
        designSpace.optimizableEdgeIds, model.safeArrayLengthsMm, ...
        frequencyData.bandRanges);
end

function seeds = generateRandomSeeds(cellCount, cellSizeMm, seedCountRange, randomStream)
%GENERATERANDOMSEEDS Generate uniformly distributed random seeds in every array cell.

    nx = cellCount(1);
    ny = cellCount(2);
    nz = cellCount(3);
    minimumSeedCount = seedCountRange(1);
    maximumSeedCount = seedCountRange(2);
    totalBlocks = prod(cellCount);
    seedsCell = cell(totalBlocks, 1);
    count = 1;
    for iz = 0:(nz-1)
        for iy = 0:(ny-1)
            for ix = 0:(nx-1)
                numSeeds = randi(randomStream, [minimumSeedCount, maximumSeedCount]);
                if numSeeds > 0
                    localPts = rand(randomStream, numSeeds, 3)*cellSizeMm;
                    offset = [ix, iy, iz]*cellSizeMm;
                    seedsCell{count} = localPts + repmat(offset, numSeeds, 1);
                end
                count = count + 1;
            end
        end
    end
    seeds = cell2mat(seedsCell(~cellfun(@isempty, seedsCell)));
end

function [cent, vol] = computeCentroidAndVolume(Vk)
    % Estimate the centroid and volume by tetrahedral decomposition.
    try
        [kTri, vol] = convhull(Vk);
        p0 = mean(Vk, 1);
        cent = [0 0 0]; volCheck = 0;
        for t = 1:size(kTri, 1)
            p1 = Vk(kTri(t,1), :); p2 = Vk(kTri(t,2), :); p3 = Vk(kTri(t,3), :);
            mat = [p1-p0; p2-p0; p3-p0];
            volTet = abs(det(mat)) / 6;
            cent = cent + volTet * ((p0 + p1 + p2 + p3) / 4);
            volCheck = volCheck + volTet;
        end
        if volCheck > 0, cent = cent / volCheck; else, cent = p0; end
    catch
        cent = mean(Vk, 1); vol = 0;
    end
end

function [area, cent, normal, orderedVids] = computePolygonProperties(pts, vids)
    % Project a planar polygon with PCA, then recover its order, area, and normal.
    try
        mu = mean(pts, 1); centered = pts - mu;
        [~, score, latent] = pca(centered);
        if latent(2) < 1e-6
            area = 0; cent = mu; normal = [0 0 1]; orderedVids = vids; return;
        end
        pts2d = score(:, 1:2);
        k = convhull(pts2d); k(end) = [];
        orderedVids = vids(k); orderedPts = pts(k, :);
        cent = mean(orderedPts, 1); area = 0; nSum = [0 0 0];
        for i = 1:length(k)
            p1 = orderedPts(i, :); p2 = orderedPts(mod(i, length(k))+1, :);
            triN = cross(p1 - cent, p2 - cent);
            area = area + 0.5 * norm(triN);
            nSum = nSum + triN;
        end
        normal = nSum / norm(nSum);
    catch
        area = 0; cent = mean(pts, 1); normal = [0 0 1]; orderedVids = vids;
    end
end

function [polygonsStruct, cavitiesStruct, V] = buildTopologyStructures(V, C, zMax, isBidirectional)
    % Extract shared Voronoi faces from vertex intersections without coplanar merging.
    polygonsStruct = struct('ID', {}, 'NodeIDs', {}, 'SharedByCavities', {}, 'Type', {}, 'Centroid', {}, 'Area', {}, 'Normal', {});
    cavitiesStruct = struct('ID', {}, 'PolygonIDs', {}, 'Volume', {}, 'NodeCoord', {});

    validC = {}; cavCount = 0;
    for k = 1:length(C)
        if isempty(C{k}) || length(C{k}) < 4, continue; end
        try
            [~, vol] = convhull(V(C{k}, :));
            if vol < 1e-6, continue; end
        catch
            continue;
        end
        cavCount = cavCount + 1;
        validC{cavCount} = C{k}; 
        cavitiesStruct(cavCount).ID = cavCount;
        cavitiesStruct(cavCount).PolygonIDs = [];
        cavitiesStruct(cavCount).Volume = vol;
        [cent, ~] = TpIMFunctions.computeCentroidAndVolume(V(C{k}, :));
        cavitiesStruct(cavCount).NodeCoord = cent;
    end

    numCells = cavCount;
    polyCount = 0;

    for i = 1:numCells
        for j = (i+1):numCells
            sharedVids = intersect(validC{i}, validC{j});
            if length(sharedVids) >= 3
                pts = V(sharedVids, :);
                [area, cent, normal, orderedVids] = TpIMFunctions.computePolygonProperties(pts, sharedVids);
                if area > 1e-6
                    polyCount = polyCount + 1;
                    polygonsStruct(polyCount).ID = polyCount;
                    polygonsStruct(polyCount).NodeIDs = orderedVids;
                    polygonsStruct(polyCount).SharedByCavities = [i, j];
                    polygonsStruct(polyCount).Type = 'Internal';
                    polygonsStruct(polyCount).Centroid = cent;
                    polygonsStruct(polyCount).Area = area;
                    dirIj = cavitiesStruct(j).NodeCoord - cavitiesStruct(i).NodeCoord;
                    if dot(normal, dirIj) < 0, normal = -normal; end
                    polygonsStruct(polyCount).Normal = normal;
                    cavitiesStruct(i).PolygonIDs = [cavitiesStruct(i).PolygonIDs, polyCount];
                    cavitiesStruct(j).PolygonIDs = [cavitiesStruct(j).PolygonIDs, polyCount];
                end
            end
        end
    end

    xMin = min(V(:,1)); xMax = max(V(:,1));
    yMin = min(V(:,2)); yMax = max(V(:,2));
    zMin = min(V(:,3));
    for i = 1:numCells
        vids = validC{i}; pts = V(vids, :); tol = 1e-3;
        planes = {
            find(abs(pts(:,1) - xMin) < tol), 'VirtualBoundary';
            find(abs(pts(:,1) - xMax) < tol), 'VirtualBoundary';
            find(abs(pts(:,2) - yMin) < tol), 'VirtualBoundary';
            find(abs(pts(:,2) - yMax) < tol), 'VirtualBoundary';
            find(abs(pts(:,3) - zMin) < tol), 'Bottom';
            find(abs(pts(:,3) - zMax) < tol), 'Top'};
        for pIdx = 1:6
            faceLocalIdx = planes{pIdx, 1}; bType = planes{pIdx, 2};
            if length(faceLocalIdx) >= 3
                faceVids = vids(faceLocalIdx); facePts = V(faceVids, :);
                [area, cent, normal, orderedVids] = TpIMFunctions.computePolygonProperties(facePts, faceVids);
                if area > 1e-6
                    polyCount = polyCount + 1;
                    polygonsStruct(polyCount).ID = polyCount;
                    polygonsStruct(polyCount).NodeIDs = orderedVids;
                    polygonsStruct(polyCount).SharedByCavities = i;
                    if strcmp(bType, 'Top')
                        polygonsStruct(polyCount).Type = 'IncidenceTop';
                    elseif strcmp(bType, 'Bottom') && isBidirectional
                        polygonsStruct(polyCount).Type = 'IncidenceBottom';
                    else
                        polygonsStruct(polyCount).Type = 'VirtualBoundary';
                    end
                    if dot(normal, cent - cavitiesStruct(i).NodeCoord) < 0, normal = -normal; end
                    polygonsStruct(polyCount).Centroid = cent;
                    polygonsStruct(polyCount).Area = area;
                    polygonsStruct(polyCount).Normal = normal;
                    cavitiesStruct(i).PolygonIDs = [cavitiesStruct(i).PolygonIDs, polyCount];
                end
            end
        end
    end
end

function [fluidVerts, fluidFacesCell, realVol, validAreasMap] = computeFluidCavityHalfspace(polyIds, polygonsStruct, tWall, centCoord, tolMerge)
    % Clip half-spaces to obtain the fluid cavity after wall-thickness offsets.
    M = length(polyIds);
    validAreasMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
    fluidVerts = centCoord; fluidFacesCell = {}; realVol = 0;
    if M < 4, return; end

    aAll = zeros(M, 3); bAll = zeros(M, 1);
    for p = 1:M
        poly = polygonsStruct(polyIds(p)); n = poly.Normal;
        if dot(n, poly.Centroid - centCoord) < 0, n = -n; end
        dOffset = tWall / 2;
        if contains(poly.Type, 'Incidence') || strcmp(poly.Type, 'VirtualBoundary')
            dOffset = 0;
        end
        aAll(p, :) = n;
        bAll(p) = dot(n, poly.Centroid) - dOffset;
    end

    validPts = [];
    triplets = nchoosek(1:M, 3);
    for k = 1:size(triplets, 1)
        idx = triplets(k, :); aSub = aAll(idx, :);
        if rcond(aSub) > 1e-4
            v = aSub \ bAll(idx);
            if all(aAll * v <= bAll + 1e-5), validPts = [validPts; v']; end 
        end
    end
    if size(validPts, 1) < 4, return; end

    cleanVerts = uniquetol(validPts, tolMerge, 'ByRows', true, 'DataScale', 1);
    try
        [kFluid, fluidVol] = convhull(cleanVerts);
    catch
        jitterSeed = 7919 + size(cleanVerts, 1);
        jitterStream = RandStream('mt19937ar', 'Seed', jitterSeed);
        jitter = (rand(jitterStream, size(cleanVerts)) - 0.5) * 1e-6;
        try
            [kFluid, fluidVol] = convhull(cleanVerts + jitter);
        catch
            return;
        end
    end

    fluidVerts = cleanVerts; realVol = fluidVol;
    numTris = size(kFluid, 1); triPlaneIdx = zeros(numTris, 1);
    for t = 1:numTris
        p1 = cleanVerts(kFluid(t,1), :); p2 = cleanVerts(kFluid(t,2), :); p3 = cleanVerts(kFluid(t,3), :);
        triCent = (p1 + p2 + p3) / 3;
        dists = abs(aAll * triCent' - bAll);
        [minDist, minIdx] = min(dists);
        if minDist < 1e-3, triPlaneIdx(t) = minIdx; else, triPlaneIdx(t) = -1; end
    end

    allUniqueIds = unique(triPlaneIdx);
    for pIdx = 1:length(allUniqueIds)
        currId = allUniqueIds(pIdx);
        triList = find(triPlaneIdx == currId);
        if isempty(triList), continue; end
        allEdges = [kFluid(triList, 1), kFluid(triList, 2); kFluid(triList, 2), kFluid(triList, 3); kFluid(triList, 3), kFluid(triList, 1)];
        sortEdges = sort(allEdges, 2);
        [~, ~, icEdge] = unique(sortEdges, 'rows');
        edgeCounts = accumarray(icEdge, 1);
        boundaryEdges = allEdges(edgeCounts(icEdge) == 1, :);
        if isempty(boundaryEdges), continue; end

        orderedVids = zeros(size(boundaryEdges, 1), 1);
        currNode = boundaryEdges(1, 1);
        validCount = 0;
        for e = 1:size(boundaryEdges, 1)
            orderedVids(e) = currNode;
            rowIdx = find(boundaryEdges(:, 1) == currNode, 1);
            if isempty(rowIdx)
                rowIdx = find(boundaryEdges(:, 2) == currNode, 1);
                if isempty(rowIdx), break; end
                currNode = boundaryEdges(rowIdx, 1);
            else
                currNode = boundaryEdges(rowIdx, 2);
            end
            boundaryEdges(rowIdx, :) = [-1, -1];
            validCount = validCount + 1;
        end

        if validCount >= 3
            orderedVidsClean = orderedVids(1:validCount)';
            fluidFacesCell{end+1} = orderedVidsClean; 
            if currId > 0
                pts = cleanVerts(orderedVidsClean, :);
                centPoly = mean(pts, 1); polyArea = 0;
                for j = 1:length(orderedVidsClean)
                    p1 = pts(j, :); p2 = pts(mod(j, length(orderedVidsClean))+1, :);
                    polyArea = polyArea + 0.5 * norm(cross(p1 - centPoly, p2 - centPoly));
                end
                origPolyId = polyIds(currId);
                validAreasMap(origPolyId) = polyArea;
            end
        end
    end
end

function [finalNodes, numNodes, finalVols, edgesIdx, lGeomArray, sFaceArray, edgeType, idxIncTop, idxIncBot, polygonsStructOut, nMaxArray, lSafeArray] = extractTpIMMatrixBounds(cavitiesStruct, polygonsStruct, vNodes, nx, ny, nz, a, hAir, areaThres, tWallMm, minBLimit, isBidirectional)
    % Build the TpIM network and bound the square-array side count on each face.
    % Update fields individually so heterogeneous structure fields remain valid.
    numCavities = length(cavitiesStruct);
    idxIncTop = 1;
    if isBidirectional
        numNodes = numCavities + 2;
        idxIncBot = numNodes;
    else
        numNodes = numCavities + 1;
        idxIncBot = -1;
    end

    finalNodes = zeros(numNodes, 3); finalVols = zeros(numNodes, 1);
    finalNodes(idxIncTop, :) = [nx*a/2, ny*a/2, nz*a + hAir/2];
    finalVols(idxIncTop) = (nx*a) * (ny*a) * hAir;
    if isBidirectional
        finalNodes(idxIncBot, :) = [nx*a/2, ny*a/2, -hAir/2];
        finalVols(idxIncBot) = (nx*a) * (ny*a) * hAir;
    end
    for i = 1:numCavities
        finalNodes(i+1, :) = cavitiesStruct(i).NodeCoord;
        finalVols(i+1) = cavitiesStruct(i).RealVolume;
    end

    edgesIdx = []; edgeType = []; sFaceArray = []; lGeomArray = [];
    nMaxArray = []; lSafeArray = [];

    % Predeclare shared fields before updating individual polygon entries.
    [polygonsStruct.nSide]  = deal(0);   % Square-array side count used by exporters.
    [polygonsStruct.n1]      = deal(0);
    [polygonsStruct.n2]      = deal(0);
    [polygonsStruct.nHoles] = deal(0);
    [polygonsStruct.nMax]   = deal(0);
    [polygonsStruct.lSafe]  = deal(0);
    [polygonsStruct.rHole]  = deal(0);
    [polygonsStruct.bSpace] = deal(0);

    for p = 1:length(polygonsStruct)
        c1 = polygonsStruct(p).SharedByCavities(1);
        c2 = 0;
        if length(polygonsStruct(p).SharedByCavities) == 2
            c2 = polygonsStruct(p).SharedByCavities(2);
        end

        S1 = 0; S2 = 0;
        if isKey(cavitiesStruct(c1).ValidFacesAreaMap, polygonsStruct(p).ID)
            S1 = cavitiesStruct(c1).ValidFacesAreaMap(polygonsStruct(p).ID);
        end
        if c2 > 0 && isKey(cavitiesStruct(c2).ValidFacesAreaMap, polygonsStruct(p).ID)
            S2 = cavitiesStruct(c2).ValidFacesAreaMap(polygonsStruct(p).ID);
        end
        realS = (S1 + S2) / (1 + (c2 > 0));
        isActiveBoundary = strcmp(polygonsStruct(p).Type, 'Internal') || contains(polygonsStruct(p).Type, 'Incidence');

        nMax = 0; lSafe = 0;
        if realS >= areaThres && isActiveBoundary
            pts = vNodes(polygonsStruct(p).NodeIDs, :);
            cent = polygonsStruct(p).Centroid;
            dMin = inf;
            numPts = size(pts, 1);
            for j = 1:numPts
                p1 = pts(j, :); p2 = pts(mod(j, numPts)+1, :);
                lVec = p2 - p1;
                if norm(lVec) < 1e-9, continue; end
                d = norm(cross(lVec, p1 - cent)) / norm(lVec);
                if d < dMin, dMin = d; end
            end
            rIn = max(0, dMin - tWallMm/2);
            lSafe = sqrt(2) * rIn;
            nMax = floor(lSafe / minBLimit);
        end

        polygonsStruct(p).nMax = nMax;
        polygonsStruct(p).lSafe = lSafe;
        if nMax == 0 || realS < areaThres || ~isActiveBoundary
            continue;
        end

        if strcmp(polygonsStruct(p).Type, 'Internal')
            distT = norm(finalNodes(c1+1,:) - polygonsStruct(p).Centroid) + norm(finalNodes(c2+1,:) - polygonsStruct(p).Centroid);
            edgesIdx = [edgesIdx; c1+1, c2+1, p]; 
            edgeType = [edgeType; 2]; 
        elseif strcmp(polygonsStruct(p).Type, 'IncidenceTop')
            distT = norm(finalNodes(idxIncTop,:) - polygonsStruct(p).Centroid) + norm(polygonsStruct(p).Centroid - finalNodes(c1+1,:));
            edgesIdx = [edgesIdx; idxIncTop, c1+1, p]; 
            edgeType = [edgeType; 3]; 
        elseif strcmp(polygonsStruct(p).Type, 'IncidenceBottom') && isBidirectional
            distT = norm(finalNodes(idxIncBot,:) - polygonsStruct(p).Centroid) + norm(polygonsStruct(p).Centroid - finalNodes(c1+1,:));
            edgesIdx = [edgesIdx; idxIncBot, c1+1, p]; 
            edgeType = [edgeType; 4]; 
        else
            continue;
        end

        sFaceArray = [sFaceArray; realS]; 
        lGeomArray = [lGeomArray; distT]; 
        nMaxArray = [nMaxArray; nMax]; 
        lSafeArray = [lSafeArray; lSafe]; 
    end
    polygonsStructOut = polygonsStruct;
end

function [nSideArray, rArray, bArray] = decodeGenes(x, W, numEdges, optIndices, lSafeArray)
    % Safely decode GA genes [r,b,N] into full acoustic-network arrays.
    % Gene r/b values use millimetres; returned r/b values use metres. N is
    % the square-array side count, so the total aperture count is N^2.
    nSideArray = zeros(numEdges, 1);
    rArray = zeros(numEdges, 1);
    bArray = zeros(numEdges, 1);
    for i = 1:W
        gId = optIndices(i);
        rRaw = max(x(i), 0);
        bRaw = max(x(W+i), 0);
        nRaw = x(2*W+i);

        bSafe = max(bRaw, 2*rRaw + 0.5);  % Prevent overlap; dimensions are in mm.
        nMax = floor(lSafeArray(gId) / bSafe);
        nSafe = max(0, min(round(nRaw), nMax));

        rArray(gId) = rRaw * 1e-3;
        bArray(gId) = bSafe * 1e-3;
        nSideArray(gId) = nSafe;
    end
end
function idx = getFrequencyBandIndices(f, band, includeRightEndpoint)
    % Return samples in [fStart,fEnd), optionally including the right endpoint.
    if numel(band) < 2 || band(1) >= band(2)
        error('TpIMFunctions:InvalidBand', ...
            'A frequency band must be [start, stop] with start < stop.');
    end
    if includeRightEndpoint
        idx = find(f >= band(1) & f <= band(2));
    else
        idx = find(f >= band(1) & f < band(2));
    end
end

function cost = calculateWeightedFitness(xSub, subIndices, xFullBase, W, fGlobal, omegaGlobal, bandRanges, wLow, wMid, wHigh, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray)
    % Evaluate the weighted low-, middle-, and high-frequency stage loss.
    xEval = xFullBase;
    xEval(subIndices) = xSub;
    numTotalEdges = size(edgesIdx, 1);
    [cNSideArr, cRArr, cBArr] = TpIMFunctions.decodeGenes(xEval, W, numTotalEdges, optIndices, lSafeArray);
    [alphaFwd, alphaBwd, ~, ~] = TpIMFunctions.solveAcoustics(cNSideArr, cRArr, cBArr, fGlobal, omegaGlobal, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot);
    alphaTarget = alphaFwd;
    if isBidirectional, alphaTarget = (alphaFwd + alphaBwd) / 2; end
    idxLow  = TpIMFunctions.getFrequencyBandIndices(fGlobal, bandRanges.low,  false);
    idxMid  = TpIMFunctions.getFrequencyBandIndices(fGlobal, bandRanges.mid,  false);
    idxHigh = TpIMFunctions.getFrequencyBandIndices(fGlobal, bandRanges.high, true);

    % Reject a stage band with no samples instead of propagating mean([]) as NaN.
    if isempty(idxLow) || isempty(idxMid) || isempty(idxHigh)
        error('TpIMFunctions:EmptyBand', ...
            'The low, middle, and high bands must overlap the global frequency grid.');
    end

    cost = -(wLow * mean(alphaTarget(idxLow)) + wMid * mean(alphaTarget(idxMid)) + wHigh * mean(alphaTarget(idxHigh)));
end

function cost = calculateGlobalFitness(x, W, f, omega, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray)
    % Minimize the negative mean absorption over the global frequency band.
    numTotalEdges = size(edgesIdx, 1);
    [cNSideArr, cRArr, cBArr] = TpIMFunctions.decodeGenes(x, W, numTotalEdges, optIndices, lSafeArray);
    [alphaFwd, alphaBwd, ~, ~] = TpIMFunctions.solveAcoustics(cNSideArr, cRArr, cBArr, f, omega, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot);
    if isBidirectional
        cost = -(mean(alphaFwd) + mean(alphaBwd)) / 2;
    else
        cost = -mean(alphaFwd);
    end
end

function [alphaFwd, alphaBwd, tlFwd, tlBwd, zNormFwd] = solveAcoustics(fullNSideArray, rArrayM, bArrayM, f, omega, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot)
    % Solve the Kirchhoff network with N^2 apertures per active square array.
    Z0 = rho0 * c;
    alphaFwd = zeros(size(f)); alphaBwd = zeros(size(f));
    tlFwd = inf(size(f)); tlBwd = inf(size(f));
    zNormFwd = zeros(size(f));
    for fi = 1:length(f)
        Y = sparse(numNodes, numNodes);
        yCavArray = 1i * omega(fi) * finalVolsM3 / (rho0 * c^2);
        for j = 1:numNodes
            Y(j, j) = yCavArray(j) + 1e-12;
        end
        for k = 1:size(edgesIdx, 1)
            nHoles = fullNSideArray(k)^2;
            if nHoles == 0, continue; end
            na = edgesIdx(k, 1); nb = edgesIdx(k, 2);
            sFace = sFaceM2(k); lBg = lGeomM(k); type = edgeType(k);
            rM = rArrayM(k); bM = bArrayM(k);
            if type == 3 || type == 4
                lAir = max(0, lBg - tTopM);
                zMpp = TpIMFunctions.calculateClusteredMppImpedance(omega(fi), nHoles, rM, bM, sFace, tTopM, rho0, c, eta, true);
            elseif type == 2
                lAir = max(0, lBg - tWallM);
                zMpp = TpIMFunctions.calculateClusteredMppImpedance(omega(fi), nHoles, rM, bM, sFace, tWallM, rho0, c, eta, false);
            else
                continue;
            end
            yBranch = 1 / (1i * omega(fi) * rho0 * lAir / sFace + zMpp);
            Y(na, nb) = Y(na, nb) - yBranch;
            Y(nb, na) = Y(nb, na) - yBranch;
            Y(na, na) = Y(na, na) + yBranch;
            Y(nb, nb) = Y(nb, nb) + yBranch;
        end

        yFwd = Y;
        if isBidirectional
            yFwd(idxIncBot, idxIncBot) = yFwd(idxIncBot, idxIncBot) + sTotalM2 / Z0;
        end
        iIn = zeros(numNodes, 1); iIn(idxIncTop) = 1;
        vFwd = yFwd \ iIn;
        zInFwd = vFwd(idxIncTop) * sTotalM2;
        rFwd = (zInFwd - Z0) / (zInFwd + Z0);
        alphaFwd(fi) = 1 - abs(rFwd)^2;
        zNormFwd(fi) = zInFwd / Z0;

        if isBidirectional && idxIncBot > 0
            yBwd = Y;
            yBwd(idxIncTop, idxIncTop) = yBwd(idxIncTop, idxIncTop) + sTotalM2 / Z0;
            iBot = zeros(numNodes, 1); iBot(idxIncBot) = 1;
            vBwd = yBwd \ iBot;
            zInBwd = vBwd(idxIncBot) * sTotalM2;
            rBwd = (zInBwd - Z0) / (zInBwd + Z0);
            alphaBwd(fi) = 1 - abs(rBwd)^2;
        end
    end
end

function zAcTotal = calculateClusteredMppImpedance(omega, nHoles, rHole, bDist, sCavity, tPlate, rho0, c, eta, isOuterPlate)
    % Return the equivalent acoustic impedance of an nHoles aperture cluster.
    sHoleTotal = nHoles * pi * rHole^2;
    sPatch = max(nHoles * (bDist^2), sHoleTotal);
    sPatch = min(sPatch, sCavity);
    if (sPatch / sCavity) >= 0.85, sPatch = sCavity; end
    phiLocal = sHoleTotal / sPatch;
    dHole = 2 * rHole;
    if phiLocal <= 1e-6, zAcTotal = Inf; return; end
    if phiLocal > 0.99
        zSpecMaa = 1i * omega * rho0 * tPlate;
    else
        kC = dHole * sqrt(rho0 * omega / (4 * eta));
        if phiLocal <= 0.08
            a = [1, -1.4092, 0, 0.33818, 0, 0.06793, -0.02287, 0.063015, -0.01614];
            Lambda = sum(a .* (sqrt(phiLocal).^(0:8)));
        else
            Lambda = max(0, 1 - 1.25 * sqrt(phiLocal));
        end
        mu = eta;
        kv = sqrt(-1i * omega * rho0 / mu);
        arg = kv * dHole / 2;
        Yv = besselj(2, arg) / besselj(0, arg);
        term = (1i * omega / (c * phiLocal)) * (tPlate + 8*dHole*Lambda/(3*pi)) / Yv;
        zBase = -term * (rho0 * c);
        if kC > 5 || dHole > 1.5e-3
            zSpecMaa = zBase + (rho0 * omega^2 * (dHole/2)^2) / (2 * c * phiLocal);
        else
            zSpecMaa = zBase;
        end
    end
    zAcMicro = zSpecMaa / sPatch;
    PhiMacro = sPatch / sCavity;
    if PhiMacro >= 0.99
        zAcMacro = 0;
    else
        rPatch = sqrt(sPatch / pi);
        LambdaMacro = max(0, 1 - 1.25 * sqrt(PhiMacro));
        if isOuterPlate, radiationFactor = 1 + LambdaMacro; else, radiationFactor = 2 * LambdaMacro; end
        hfRelaxMacro = 1.0 / (1 + ((omega/c) * rPatch)^2);
        lMacro = (8 * rPatch / (3 * pi)) * radiationFactor * sqrt(phiLocal) * hfRelaxMacro;
        zAcMacro = 1i * omega * rho0 * lMacro / sPatch;
    end
    zAcTotal = zAcMicro + zAcMacro;
end

function [cavGroups, faceGroups, membership] = classifyCavities(method, cavitiesStruct, edgesIdx, finalNodes, optIndices, W, renderDiagnostics)
    % Group cavities from depth, volume, and connectivity, then map groups to faces.
    if nargin < 7
        renderDiagnostics = true;
    end
    numCavs = length(cavitiesStruct);
    zTop = max(finalNodes(:,3));
    fMat = zeros(numCavs, 3);
    for i = 1:numCavs
        fMat(i,1) = cavitiesStruct(i).RealVolume;
        fMat(i,2) = zTop - cavitiesStruct(i).NodeCoord(3);
        fMat(i,3) = sum(edgesIdx(:,1)==(i+1) | edgesIdx(:,2)==(i+1));
    end
    fNorm = (fMat - min(fMat)) ./ (max(fMat) - min(fMat) + 1e-9);
    idxCluster = zeros(numCavs, 1);
    membership = zeros(numCavs, 3);

    switch lower(method)
        case 'spectral'
            A = sparse(numCavs, numCavs);
            for k = 1:size(edgesIdx, 1)
                c1 = edgesIdx(k, 1) - 1; c2 = edgesIdx(k, 2) - 1;
                if c1 > 0 && c2 > 0 && c1 <= numCavs && c2 <= numCavs
                    A(c1, c2) = 1; A(c2, c1) = 1;
                end
            end
            dMat = diag(sum(A, 2));
            L = dMat - A;
            try
                [EigVec, ~] = eigs(L + 1e-6*speye(numCavs), 3, 'smallestabs');
                rng(42); idxCluster = kmeans(EigVec, 3, 'Replicates', 5);
            catch
                rng(42); idxCluster = kmeans(fNorm(:,2), 3);
            end
            for i = 1:numCavs, membership(i, idxCluster(i)) = 1; end

        case 'modal'
            localOmegaProxy = zeros(numCavs, 1);
            for i = 1:numCavs
                cI = fMat(i,1);
                mI = 1.0 / max(1, fMat(i,3));
                localOmegaProxy(i) = 1.0 / sqrt(cI * mI + 1e-9);
            end
            [~, sortIdx] = sort(localOmegaProxy, 'ascend');
            binSize = ceil(numCavs / 3);
            idxCluster(sortIdx(1:binSize)) = 1;
            idxCluster(sortIdx(binSize+1:min(2*binSize, numCavs))) = 2;
            if 2*binSize+1 <= numCavs
                idxCluster(sortIdx(2*binSize+1:end)) = 3;
            end
            idxCluster(idxCluster == 0) = 3;
            for i = 1:numCavs, membership(i, idxCluster(i)) = 1; end

        case 'fuzzy'
            options = [2.0, 100, 1e-5, 0];
            [~, U] = fcm(fNorm(:, 1:2), 3, options);
            membership = U';
            [~, idxCluster] = max(membership, [], 2);

        otherwise
            error('TpIMFunctions:UnknownClusteringMethod', ...
                'Unknown cavity-classification method: %s.', method);
    end

    meanDepth = zeros(1, 3);
    for g = 1:3
        if any(idxCluster == g)
            meanDepth(g) = mean(fMat(idxCluster == g, 2));
        else
            meanDepth(g) = inf;
        end
    end
    [~, depthOrder] = sort(meanDepth, 'ascend');

    cavGroups = zeros(size(idxCluster));
    cavGroups(idxCluster == depthOrder(1)) = 3; % Shallow: high-frequency group.
    cavGroups(idxCluster == depthOrder(2)) = 2; % Intermediate: middle-frequency group.
    cavGroups(idxCluster == depthOrder(3)) = 1; % Deep: low-frequency group.

    membershipAligned = zeros(size(membership));
    membershipAligned(:, 3) = membership(:, depthOrder(1));
    membershipAligned(:, 2) = membership(:, depthOrder(2));
    membershipAligned(:, 1) = membership(:, depthOrder(3));
    membership = membershipAligned;

    faceGroups = zeros(W, 1);
    for i = 1:W
        c1 = edgesIdx(optIndices(i), 1) - 1;
        c2 = edgesIdx(optIndices(i), 2) - 1;
        g1 = 1; if c1 > 0 && c1 <= numCavs, g1 = cavGroups(c1); end
        g2 = 1; if c2 > 0 && c2 <= numCavs, g2 = cavGroups(c2); end
        if c1 == 0
            faceGroups(i) = g2;
        elseif c2 == 0
            faceGroups(i) = g1;
        else
            faceGroups(i) = max(g1, g2);
        end
    end

    if ~renderDiagnostics
        return;
    end
    colors = {[0.85 0.2 0.2], [0.2 0.8 0.2], [0.2 0.5 0.9]};
    figure('Name', ['Feature Space Clustering - ', upper(method)], 'Color', 'w');
    hold on; grid on; box on; view(45, 30);
    for g = 1:3
        idx = find(cavGroups == g);
        if ~isempty(idx)
            scatter3(fNorm(idx, 2), fNorm(idx, 1), fNorm(idx, 3), 120, ...
                'MarkerFaceColor', colors{g}, 'MarkerFaceAlpha', 1.0, ...
                'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
        end
    end
    title(sprintf('Clustering Feature Space (%s)', upper(method)), 'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    xlabel('Normalized depth (shallow \rightarrow deep)', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('Normalized volume (small \rightarrow large)', 'FontSize', 12, 'FontName', 'Times New Roman');
    zlabel('Connectivity degree (low \rightarrow high)', 'FontSize', 12, 'FontName', 'Times New Roman');
    xlim([-0.05 1.05]); ylim([-0.05 1.05]); zlim([-0.05 1.05]);
    set(gca, 'FontSize', 11, 'FontName', 'Times New Roman', 'GridAlpha', 0.3);
    legend({'Low Freq (Deep Vault)', 'Mid Freq (Transition)', 'High Freq (Shallow Gateway)'}, 'Location', 'best', 'FontSize', 11, 'FontName', 'Times New Roman');
end

function renderClusteredCavities(cavitiesStruct, finalNodes, edgesIdx, optIndices, cavGroups, faceGroups, displayShrink)
    % Render shrunken cavities and frequency-group connection paths.
    colors = {[0.85 0.2 0.2], [0.2 0.8 0.2], [0.2 0.5 0.9]};
    hold on;
    for i = 1:length(cavitiesStruct)
        vertsOrig = cavitiesStruct(i).FluidVerts;
        facesCell = cavitiesStruct(i).FluidFaces;
        cent = cavitiesStruct(i).NodeCoord;
        if size(vertsOrig, 1) < 4 || isempty(facesCell), continue; end
        gIdx = cavGroups(i);
        if gIdx < 1 || gIdx > 3, continue; end
        cCav = colors{gIdx};
        vecs = vertsOrig - repmat(cent, size(vertsOrig, 1), 1);
        vertsShrunk = repmat(cent, size(vertsOrig, 1), 1) + vecs * displayShrink;
        for f = 1:length(facesCell)
            patch('Vertices', vertsShrunk, 'Faces', facesCell{f}, 'FaceColor', cCav, 'FaceAlpha', 1, 'EdgeColor', cCav * 0.4, 'LineWidth', 0.5);
        end
    end
    for k = 1:length(optIndices)
        gIdx = optIndices(k);
        c1 = edgesIdx(gIdx, 1); c2 = edgesIdx(gIdx, 2);
        fg = faceGroups(k);
        if fg < 1 || fg > 3, continue; end
        cLine = colors{fg};
        p1 = finalNodes(c1, :); p2 = finalNodes(c2, :);
        line([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'Color', cLine, 'LineWidth', 2, 'LineStyle','-');
    end
    view(35, 25); axis equal; axis off;
    camlight('headlight'); lighting flat;
end

function renderTopologyExplorer(polygonsStruct, cavitiesStruct, V)
    % Render the base topology and boundary categories.
    figure('Name', 'Acoustic Topology Explorer', 'Color', 'w');
    hold on; view(35, 25); grid on; axis equal; axis off;
    for p = 1:length(polygonsStruct)
        poly = polygonsStruct(p); verts = V(poly.NodeIDs, :);
        if strcmp(poly.Type, 'Internal')
            patch('Vertices', verts, 'Faces', 1:length(poly.NodeIDs), 'FaceColor', [0.1 0.7 0.1], 'FaceAlpha', 1, 'EdgeColor', 'k');
        elseif contains(poly.Type, 'Incidence')
            patch('Vertices', verts, 'Faces', 1:length(poly.NodeIDs), 'FaceColor', [1.0 0.2 0.2], 'FaceAlpha', 0, 'EdgeColor', 'k');
        elseif strcmp(poly.Type, 'VirtualBoundary')
            patch('Vertices', verts, 'Faces', 1:length(poly.NodeIDs), 'FaceColor', [0.6 0.2 0.8], 'FaceAlpha', 0, 'EdgeColor', 'k');
        end
    end
    if ~isempty(cavitiesStruct)
        nodesMat = reshape([cavitiesStruct.NodeCoord], 3, [])';
        scatter3(nodesMat(:,1), nodesMat(:,2), nodesMat(:,3), 80, 'k', 'filled', 'MarkerEdgeColor', 'w');
    end
    axis off;
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 17.82);
end

function generateScdmScript(polygonsStruct, cavitiesStruct, tWall, tTop, incH, nx, ny, nz, a, isBidirectional)
    % Export a SpaceClaim/SCDM Python macro with N-by-N square aperture arrays.
    %
    % The polyhedral air domain spans z=0 to z=nz*a. Plate channels are
    % centred through the top or bottom plate, and exterior domains begin
    % beyond the complete plate thickness.
    fileScdm = 'scdmModelHybridSquare.py';
    fid = fopen(fileScdm, 'wt');
    if fid < 0
        error('TpIMFunctions:ScdmFileCreationFailed', ...
            'Unable to create the SCDM script file.');
    end

    fprintf(fid, '# -*- coding: utf-8 -*-\n');
    fprintf(fid, 'import System\nfrom System.Collections.Generic import List\nimport math\n');
    fprintf(fid, 'ClearAll()\n');
    fprintf(fid, 'try: DocumentHelper.PauseHistoryRecording()\nexcept: pass\n\n');

    fprintf(fid, 'allBodies = []\n');
    fprintf(fid, 'def hideBody(b):\n');
    fprintf(fid, '    try:\n');
    fprintf(fid, '        ViewHelper.SetObjectVisibility(Selection.Create(b), VisibilityType.Hide, False, False)\n');
    fprintf(fid, '        allBodies.append(b)\n');
    fprintf(fid, '    except: pass\n\n');

    fprintf(fid, 'def createSquareArrayHoles(cx, cy, cz, nx, ny, nz, nSide, rHole, spacing, tLen):\n');
    fprintf(fid, '    createdBodies = []\n');
    fprintf(fid, '    if nSide <= 0: return createdBodies\n');
    fprintf(fid, '    nVec = Vector.Create(nx, ny, nz)\n');
    fprintf(fid, '    if nVec.Magnitude <= 1e-12: return createdBodies\n');
    fprintf(fid, '    nVec = Vector.Create(nVec.X/nVec.Magnitude, nVec.Y/nVec.Magnitude, nVec.Z/nVec.Magnitude)\n');
    fprintf(fid, '    if abs(nVec.X) > 0.9:\n');
    fprintf(fid, '        uVec = Vector.Create(0, 1, 0)\n');
    fprintf(fid, '    else:\n');
    fprintf(fid, '        uC = Vector.Cross(nVec, Vector.Create(1, 0, 0))\n');
    fprintf(fid, '        uVec = Vector.Create(uC.X/uC.Magnitude, uC.Y/uC.Magnitude, uC.Z/uC.Magnitude)\n');
    fprintf(fid, '    vC = Vector.Cross(nVec, uVec)\n');
    fprintf(fid, '    vVec = Vector.Create(vC.X/vC.Magnitude, vC.Y/vC.Magnitude, vC.Z/vC.Magnitude)\n');
    fprintf(fid, '    offsetU = - (nSide - 1) / 2.0 * spacing\n');
    fprintf(fid, '    offsetV = - (nSide - 1) / 2.0 * spacing\n');
    fprintf(fid, '    for i in range(int(nSide)):\n');
    fprintf(fid, '        for j in range(int(nSide)):\n');
    fprintf(fid, '            dx = offsetU + i * spacing\n');
    fprintf(fid, '            dy = offsetV + j * spacing\n');
    fprintf(fid, '            pCx = cx + uVec.X * dx + vVec.X * dy\n');
    fprintf(fid, '            pCy = cy + uVec.Y * dx + vVec.Y * dy\n');
    fprintf(fid, '            pCz = cz + uVec.Z * dx + vVec.Z * dy\n');
    fprintf(fid, '            p1X = pCx - nVec.X * (tLen / 2.0)\n');
    fprintf(fid, '            p1Y = pCy - nVec.Y * (tLen / 2.0)\n');
    fprintf(fid, '            p1Z = pCz - nVec.Z * (tLen / 2.0)\n');
    fprintf(fid, '            p2X = pCx + nVec.X * (tLen / 2.0)\n');
    fprintf(fid, '            p2Y = pCy + nVec.Y * (tLen / 2.0)\n');
    fprintf(fid, '            p2Z = pCz + nVec.Z * (tLen / 2.0)\n');
    fprintf(fid, '            p3X = p2X + uVec.X * rHole\n');
    fprintf(fid, '            p3Y = p2Y + uVec.Y * rHole\n');
    fprintf(fid, '            p3Z = p2Z + uVec.Z * rHole\n');
    fprintf(fid, '            try:\n');
    fprintf(fid, '                res = CylinderBody.Create(Point.Create(MM(p1X), MM(p1Y), MM(p1Z)), Point.Create(MM(p2X), MM(p2Y), MM(p2Z)), Point.Create(MM(p3X), MM(p3Y), MM(p3Z)), ExtrudeType.ForceIndependent)\n');
    fprintf(fid, '                for b in res.CreatedBodies: createdBodies.append(b)\n');
    fprintf(fid, '            except: pass\n');
    fprintf(fid, '    return createdBodies\n\n');

    fprintf(fid, '# --- Direct Additive Fluid Domain ---\n');
    for i = 1:length(cavitiesStruct)
        facesCell = cavitiesStruct(i).FluidFaces;
        verts = cavitiesStruct(i).FluidVerts;
        if isempty(facesCell), continue; end
        fprintf(fid, 'cavitySurfs = List[IDesignBody]()\n');
        for f = 1:length(facesCell)
            nodes = facesCell{f};
            for j = 1:length(nodes)
                p1 = verts(nodes(j), :);
                p2 = verts(nodes(mod(j, length(nodes))+1), :);
                fprintf(fid, 'SketchLine.Create(Point.Create(MM(%f),MM(%f),MM(%f)), Point.Create(MM(%f),MM(%f),MM(%f)))\n', p1(1), p1(2), p1(3), p2(1), p2(2), p2(3));
            end
            fprintf(fid, 'curvelist = List[IDesignCurve]()\nfor c in GetRootPart().Curves: curvelist.Add(c)\n');
            fprintf(fid, 'try:\n    Fill.Execute(Selection.Create(curvelist))\n    cavitySurfs.Add(GetRootPart().Bodies[-1])\nexcept: pass\nDelete.Execute(Selection.Create(curvelist))\n');
        end
        fprintf(fid, 'if cavitySurfs.Count > 0:\n    try:\n        Combine.Merge(Selection.Create(cavitySurfs))\n        hideBody(GetRootPart().Bodies[-1])\n    except: pass\n\n');
    end

    fprintf(fid, '# --- Connecting Necks: square IPP array ---\n');
    for p = 1:length(polygonsStruct)
        poly = polygonsStruct(p);
        nSide = 0;
        if isfield(poly, 'nSide'), nSide = poly.nSide; end
        if nSide > 0 && ~strcmp(poly.Type, 'VirtualBoundary')
            cent = poly.Centroid;
            normal = poly.Normal;
            if norm(normal) > 1e-12
                normal = normal ./ norm(normal);
            end

            % Internal channels remain centred on shared polyhedral faces.
            % Incidence channels are shifted to the plate mid-plane and
            % extended slightly into adjacent domains to support merging.
            tLen = tWall * 1.5;
            if strcmp(poly.Type, 'IncidenceTop')
                cent(3) = nz * a + tTop / 2;
                normal = [0, 0, 1];
                tLen = tTop * 1.5;
            elseif strcmp(poly.Type, 'IncidenceBottom')
                cent(3) = -tTop / 2;
                normal = [0, 0, -1];
                tLen = tTop * 1.5;
            elseif contains(poly.Type, 'Incidence')
                cent = cent + normal * (tTop / 2);
                tLen = tTop * 1.5;
            end

            rOpt = poly.rHole;
            bOpt = poly.bSpace;
            fprintf(fid, 'cylinders = createSquareArrayHoles(%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %d, %.6f, %.6f, %.6f)\n', ...
                cent(1), cent(2), cent(3), normal(1), normal(2), normal(3), nSide, rOpt, bOpt, tLen);
            fprintf(fid, 'for b in cylinders: hideBody(b)\n');
        end
    end

    fprintf(fid, '\n# --- Top Incidence & PML ---\n');
    xMax = nx * a;
    yMax = ny * a;
    zCavityTop = nz * a;
    zTopOuter = zCavityTop + tTop;
    zIncStart = zTopOuter;
    fprintf(fid, 'try:\n    res = BlockBody.Create(Point.Create(MM(0), MM(0), MM(%.6f)), Point.Create(MM(%.6f), MM(%.6f), MM(%.6f)))\n    for b in res.CreatedBodies: hideBody(b)\nexcept: pass\n', zIncStart, xMax, yMax, zIncStart + incH);
    fprintf(fid, 'try:\n    res = BlockBody.Create(Point.Create(MM(0), MM(0), MM(%.6f)), Point.Create(MM(%.6f), MM(%.6f), MM(%.6f)))\n    for b in res.CreatedBodies: hideBody(b)\nexcept: pass\n', zIncStart + incH, xMax, yMax, zIncStart + 2*incH);

    if isBidirectional
        fprintf(fid, '\n# --- Bottom Incidence & PML ---\n');
        zBottomOuter = -tTop;
        zBotStart = zBottomOuter;
        fprintf(fid, 'try:\n    res = BlockBody.Create(Point.Create(MM(0), MM(0), MM(%.6f)), Point.Create(MM(%.6f), MM(%.6f), MM(%.6f)))\n    for b in res.CreatedBodies: hideBody(b)\nexcept: pass\n', zBotStart - incH, xMax, yMax, zBotStart);
        fprintf(fid, 'try:\n    res = BlockBody.Create(Point.Create(MM(0), MM(0), MM(%.6f)), Point.Create(MM(%.6f), MM(%.6f), MM(%.6f)))\n    for b in res.CreatedBodies: hideBody(b)\nexcept: pass\n', zBotStart - 2*incH, xMax, yMax, zBotStart - incH);
    end

    fprintf(fid, 'try:\n    if len(allBodies) > 0:\n        allBSelection = Selection.Create(allBodies)\n        ViewHelper.SetObjectVisibility(allBSelection, VisibilityType.Show, False, False)\nexcept: pass\n');
    fprintf(fid, '\n# To fuse fluid domains, select them in SCDM and run Combine -> Merge.\n');
    fprintf(fid, 'try: DocumentHelper.ResumeHistoryRecording()\nexcept: pass\n');
    fprintf(fid, 'print("[DONE] Square TpIM SCDM model script finished.")\n');
    fclose(fid);
end

% =========================================================================
% Post-processing utilities for absorption, complex frequency, and 3-D fields.
% =========================================================================
function renderAbsorptionAndImpedance(fShow, alphaShow, zNormShow, bandRanges)
    % Render final one-sided absorption and normalized input impedance.
    figure('Name', 'Final Single-sided Absorption Response', 'Color', 'w');
    hold on; grid on; box on;
    fr = bandRanges.global;
    patch([fr(1) fr(2) fr(2) fr(1)], [0 0 1.05 1.05], [0.2 0.5 0.9], ...
        'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(fShow, real(alphaShow), 'b-', 'LineWidth', 2.7);
    text(mean(fr), 0.98, sprintf('Global optimization band\n%.0f-%.0f Hz', fr(1), fr(2)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'BackgroundColor', 'w', 'Margin', 5, 'FontName', 'Times New Roman');
    title('Final single-sided absorption response', 'FontSize', 14, 'FontName', 'Times New Roman');
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('Absorption coefficient \alpha', 'FontSize', 12, 'FontName', 'Times New Roman');
    xlim([min(fShow), max(fShow)]); ylim([0, 1.05]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    figure('Name', 'Final Single-sided Input Impedance', 'Color', 'w');
    hold on; grid on; box on;
    plot(fShow, real(zNormShow), 'b-', 'LineWidth', 2.0);
    plot(fShow, imag(zNormShow), 'b--', 'LineWidth', 1.8);
    yline(1, 'k:', 'Resistance = 1', 'LabelHorizontalAlignment', 'left');
    yline(0, 'k-.', 'Reactance = 0', 'LabelHorizontalAlignment', 'left');
    yMin = min([real(zNormShow(:)); imag(zNormShow(:)); 0; 1]) - 0.5;
    yMax = max([real(zNormShow(:)); imag(zNormShow(:)); 0; 1]) + 0.5;
    patch([fr(1) fr(2) fr(2) fr(1)], [yMin yMin yMax yMax], [0.2 0.5 0.9], ...
        'FaceAlpha', 0.06, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    ylim([yMin, yMax]);
    title('Normalized input impedance seen from incidence side', 'FontSize', 14, 'FontName', 'Times New Roman');
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('Normalized impedance zS/z0', 'FontSize', 12, 'FontName', 'Times New Roman');
    legend('Re(Z)', 'Im(Z)', 'Location', 'best', 'FontSize', 10);
    xlim([min(fShow), max(fShow)]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
end

function maps = renderComplexFrequencyPlane(nSideArray, rArrayM, bArrayM, cfg, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, sTotalM2, idxIncTop)
    % Evaluate the one-sided reflection map in the complex-frequency plane.
    if ~isfield(cfg, 'fRealRange'), cfg.fRealRange = [20, 1600]; end
    if ~isfield(cfg, 'fImagRange'), cfg.fImagRange = [-600, 600]; end
    if ~isfield(cfg, 'nReal'), cfg.nReal = 60; end
    if ~isfield(cfg, 'nImag'), cfg.nImag = 60; end

    fRealGrid = linspace(cfg.fRealRange(1), cfg.fRealRange(2), cfg.nReal);
    fImagGrid = linspace(cfg.fImagRange(1), cfg.fImagRange(2), cfg.nImag);
    [Fr, Fi] = meshgrid(fRealGrid, fImagGrid);
    rMap = zeros(size(Fr));

    for row = 1:size(Fr, 1)
        for col = 1:size(Fr, 2)
            cOmega = 2*pi*(Fr(row, col) + 1i*Fi(row, col));
            rMap(row, col) = TpIMFunctions.evaluateComplexReflection(cOmega, ...
                nSideArray, rArrayM, bArrayM, numNodes, finalVolsM3, ...
                edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, sTotalM2, idxIncTop);
        end
    end
    r2Map = abs(rMap).^2;

    maps.Fr = Fr;
    maps.Fi = Fi;
    maps.R = rMap;
    maps.R2 = r2Map;
    [maps.minR2, minIdx] = min(r2Map(:));
    [rIdx, cIdx] = ind2sub(size(r2Map), minIdx);
    maps.zeroLikeFrequency = [Fr(rIdx, cIdx), Fi(rIdx, cIdx)];

    figure('Name', 'Complex Frequency Plane - Single-sided Reflection Zero', 'Color', 'w');
    TpIMFunctions.renderComplexFrequencyMap(Fr, Fi, r2Map, 'Single-sided reflection zero map');
end

function rComplex = evaluateComplexReflection(cOmega, nSideArray, rArrayM, bArrayM, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, sTotalM2, idxIncTop)
    Z0 = rho0 * c;
    yComplex = sparse(numNodes, numNodes);
    yCav = 1i * cOmega * finalVolsM3 / (rho0 * c^2);
    for j = 1:numNodes
        yComplex(j, j) = yCav(j) + 1e-12;
    end

    for k = 1:size(edgesIdx, 1)
        nHoles = nSideArray(k)^2;
        if nHoles == 0, continue; end
        na = edgesIdx(k, 1); nb = edgesIdx(k, 2);
        sFace = sFaceM2(k); lBg = lGeomM(k); type = edgeType(k);
        rM = rArrayM(k); bM = bArrayM(k);
        if type == 3 || type == 4
            lAir = max(0, lBg - tTopM);
            zMpp = TpIMFunctions.calculateClusteredMppImpedance(cOmega, nHoles, rM, bM, sFace, tTopM, rho0, c, eta, true);
        elseif type == 2
            lAir = max(0, lBg - tWallM);
            zMpp = TpIMFunctions.calculateClusteredMppImpedance(cOmega, nHoles, rM, bM, sFace, tWallM, rho0, c, eta, false);
        else
            continue;
        end
        yBranch = 1 / (1i * cOmega * rho0 * lAir / sFace + zMpp);
        yComplex(na, nb) = yComplex(na, nb) - yBranch;
        yComplex(nb, na) = yComplex(nb, na) - yBranch;
        yComplex(na, na) = yComplex(na, na) + yBranch;
        yComplex(nb, nb) = yComplex(nb, nb) + yBranch;
    end

    iInC = zeros(numNodes, 1);
    iInC(idxIncTop) = 1;
    vC = yComplex \ iInC;
    zInC = vC(idxIncTop) * sTotalM2;
    rComplex = (zInC - Z0) / (zInC + Z0);
end

function renderComplexFrequencyMap(Fr, Fi, r2Map, ttl)
    contourf(Fr, Fi, log10(r2Map + 1e-8), 40, 'LineColor', 'none');
    colormap(gca, turbo(256));
    cb = colorbar;
    cb.Label.String = 'log_{10}|R|^2';
    cb.Label.FontName = 'Times New Roman';
    hold on;
    yline(0, 'w--', 'LineWidth', 1.5);
    [~, minIdx] = min(r2Map(:));
    [rIdx, cIdx] = ind2sub(size(r2Map), minIdx);
    plot(Fr(rIdx, cIdx), Fi(rIdx, cIdx), 'w*', 'MarkerSize', 9, 'LineWidth', 1.8);
    title(ttl, 'FontSize', 14, 'FontName', 'Times New Roman');
    xlabel('Real Frequency \omegaR / 2\pi (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('Imaginary Frequency \omegaI / 2\pi (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
end

function state = computeFullBandAcousticState(nSideArray, rArrayM, bArrayM, fGlobal, omegaGlobal, fViz, cavitiesStruct, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot)
    % Compute full-band nodal admittance and pressure; one-sided cases use top incidence.
    [~, idxViz] = min(abs(fGlobal - fViz));
    fVizActual = fGlobal(idxViz);
    pNodesAll = zeros(length(fGlobal), numNodes);
    yNodesAll = zeros(length(fGlobal), numNodes);
    yViz = sparse(numNodes, numNodes);
    vNodesViz = zeros(numNodes, 1);
    yBranchVizArray = zeros(size(edgesIdx, 1), 1);

    Z0 = rho0 * c;
    y0Port = sTotalM2 / Z0;
    for fi = 1:length(fGlobal)
        Y = sparse(numNodes, numNodes);
        yCavArray = 1i * omegaGlobal(fi) * finalVolsM3 / (rho0 * c^2);
        for j = 1:numNodes
            Y(j, j) = yCavArray(j) + 1e-12;
        end

        for k = 1:size(edgesIdx, 1)
            nHoles = nSideArray(k)^2;
            if nHoles == 0, continue; end
            na = edgesIdx(k, 1); nb = edgesIdx(k, 2);
            sFace = sFaceM2(k); lBg = lGeomM(k); type = edgeType(k);
            rM = rArrayM(k); bM = bArrayM(k);
            if type == 3 || type == 4
                lAir = max(0, lBg - tTopM);
                zMpp = TpIMFunctions.calculateClusteredMppImpedance(omegaGlobal(fi), nHoles, rM, bM, sFace, tTopM, rho0, c, eta, true);
            elseif type == 2
                lAir = max(0, lBg - tWallM);
                zMpp = TpIMFunctions.calculateClusteredMppImpedance(omegaGlobal(fi), nHoles, rM, bM, sFace, tWallM, rho0, c, eta, false);
            else
                continue;
            end
            yBranch = 1 / (1i * omegaGlobal(fi) * rho0 * lAir / sFace + zMpp);
            if fi == idxViz, yBranchVizArray(k) = yBranch; end
            Y(na, nb) = Y(na, nb) - yBranch;
            Y(nb, na) = Y(nb, na) - yBranch;
            Y(na, na) = Y(na, na) + yBranch;
            Y(nb, nb) = Y(nb, nb) + yBranch;
        end

        yFwd = Y;
        if isBidirectional && idxIncBot > 0
            yFwd(idxIncBot, idxIncBot) = yFwd(idxIncBot, idxIncBot) + y0Port;
        end
        iIn = zeros(numNodes, 1);
        iIn(idxIncTop) = 1;
        vFwd = yFwd \ iIn;

        pNodesAll(fi, :) = abs(vFwd);
        yNodesAll(fi, :) = abs(diag(yFwd));
        if fi == idxViz
            yViz = yFwd;
            vNodesViz = vFwd;
        end
    end

    state.fViz = fViz;
    state.idxViz = idxViz;
    state.fVizActual = fVizActual;
    state.pNodesAll = pNodesAll;
    state.yNodesAll = yNodesAll;
    state.yViz = yViz;
    state.vNodesViz = vNodesViz;
    state.yBranchVizArray = yBranchVizArray;
    state.cavityNodeIds = 2:(1 + length(cavitiesStruct));
end

function renderFullBandPhysics(fGlobal, state, cavitiesStruct)
    cavityNodeIds = state.cavityNodeIds;
    if isempty(cavityNodeIds)
        warning('TpIMFunctions:NoCavityNodes', ...
            'No physical cavity nodes are available for plotting.');
        return;
    end
    yCavAll = state.yNodesAll(:, cavityNodeIds);
    pCavAll = state.pNodesAll(:, cavityNodeIds);

    figure('Name', 'Multi-physics Frequency Responses - Cavities Only', ...
        'Color', 'w', 'Position', [100, 100, 1120, 420]);
    subplot(1,2,1); hold on; grid on; box on;
    plot(fGlobal, yCavAll, 'LineWidth', 0.8, 'Color', [0.45 0.65 0.90]);
    plot(fGlobal, mean(yCavAll, 2), 'r-', 'LineWidth', 3.0);
    title(sprintf('Optimized structure: cavity admittance (%d cavities)', ...
        length(cavitiesStruct)), 'FontSize', 13, 'FontName', 'Times New Roman');
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('Nodal admittance magnitude |Y_{ii}|', 'FontSize', 12, 'FontName', 'Times New Roman');
    set(gca, 'YScale', 'log', 'FontName', 'Times New Roman');

    subplot(1,2,2); hold on; grid on; box on;
    plot(fGlobal, pCavAll, 'LineWidth', 0.8, 'Color', [0.55 0.80 0.55]);
    plot(fGlobal, mean(pCavAll, 2), 'k-', 'LineWidth', 3.0);
    title(sprintf('Optimized structure: cavity pressure (%d cavities)', ...
        length(cavitiesStruct)), 'FontSize', 13, 'FontName', 'Times New Roman');
    xlabel('Frequency (Hz)', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('Sound pressure magnitude |P|', 'FontSize', 12, 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman');
end

function renderPolyhedralAcousticMaps(cavitiesStruct, polygonsStruct, V, finalNodes, edgesIdx, edgeType, nSideArray, sFaceArray, state, idxIncTop, idxIncBot, isBidirectional)
    % Render 3-D acoustic fields on the clipped polyhedral fluid cavities.
    cmapGlobal = turbo(256);
    activeMask = ismember(edgeType(:), [2, 3, 4]) & (nSideArray(:) > 0);
    sMag = sFaceArray(:);
    yBranchMag = abs(state.yBranchVizArray(:));
    yNodeMag = abs(diag(state.yViz));
    pMag = abs(state.vNodesViz(:));
    cavNodeIds = state.cavityNodeIds;
    yCavMag = yNodeMag(cavNodeIds);
    pCavMag = pMag(cavNodeIds);

    climS = TpIMFunctions.calculateSafeColorLimits(sMag(activeMask));
    climY = TpIMFunctions.calculateSafeColorLimits([yBranchMag(activeMask); yCavMag(:)]);
    climP = TpIMFunctions.calculateSafeColorLimits(pCavMag(:));

    figure('Name', '3D Polyhedral Acoustic State Mappings', ...
        'Color', 'k', 'Position', [100, 100, 1680, 420]);
    set(gcf, 'InvertHardcopy', 'off');
    tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile; hold on; view(35, 25); axis equal; axis off;
    set(gca, 'Color', 'k');
    title(sprintf('Active interface area S (f = %.1f Hz)', state.fVizActual), 'FontSize', 13, 'Color', 'w', 'FontName', 'Times New Roman');
    TpIMFunctions.renderFluidCavityShells(cavitiesStruct, 0.08, [0.75 0.75 0.75], [0.25 0.25 0.25]);
    TpIMFunctions.renderActiveInterfaceFaces(polygonsStruct, V, edgesIdx, activeMask, sMag, cmapGlobal, climS, 0.78);
    TpIMFunctions.renderBranchLinesByValue(finalNodes, edgesIdx, activeMask, sMag, cmapGlobal, climS, 1.0);
    colormap(gca, cmapGlobal); cb1 = colorbar; cb1.Color = 'w'; cb1.Label.String = 'Area S (mm^2)'; caxis(climS);
    TpIMFunctions.apply3DViewStyle();

    nexttile; hold on; view(35, 25); axis equal; axis off;
    set(gca, 'Color', 'k');
    title(sprintf('Cavity / branch admittance |Y| (f = %.1f Hz)', state.fVizActual), 'FontSize', 13, 'Color', 'w', 'FontName', 'Times New Roman');
    TpIMFunctions.renderFluidCavitiesByValue(cavitiesStruct, yCavMag, cmapGlobal, climY, 0.64, [0.10 0.10 0.10]);
    TpIMFunctions.renderBranchLinesByValue(finalNodes, edgesIdx, activeMask, yBranchMag, cmapGlobal, climY, 2.3);
    colormap(gca, cmapGlobal); cb2 = colorbar; cb2.Color = 'w'; cb2.Label.String = '|Y| Magnitude'; caxis(climY);
    TpIMFunctions.apply3DViewStyle();

    nexttile; hold on; view(35, 25); axis equal; axis off;
    set(gca, 'Color', 'k');
    title(sprintf('Cavity sound pressure |P| (f = %.1f Hz)', state.fVizActual), 'FontSize', 13, 'Color', 'w', 'FontName', 'Times New Roman');
    TpIMFunctions.renderFluidCavitiesByValue(cavitiesStruct, pCavMag, cmapGlobal, climP, 0.72, [0.10 0.10 0.10]);
    TpIMFunctions.renderBranchLinesByValue(finalNodes, edgesIdx, activeMask, ones(size(yBranchMag))*mean(climP), cmapGlobal, climP, 1.0);
    scatter3(finalNodes(idxIncTop,1), finalNodes(idxIncTop,2), finalNodes(idxIncTop,3), 180, 'w', 'filled', 'MarkerEdgeColor', [0.9 0.1 0.1], 'LineWidth', 2);
    text(finalNodes(idxIncTop,1), finalNodes(idxIncTop,2), finalNodes(idxIncTop,3), '  Top port', 'Color', 'w', 'FontName', 'Times New Roman');
    if isBidirectional && idxIncBot > 0
        scatter3(finalNodes(idxIncBot,1), finalNodes(idxIncBot,2), finalNodes(idxIncBot,3), 180, 'w', 'filled', 'MarkerEdgeColor', [0.1 0.3 0.9], 'LineWidth', 2);
        text(finalNodes(idxIncBot,1), finalNodes(idxIncBot,2), finalNodes(idxIncBot,3), '  Bottom port', 'Color', 'w', 'FontName', 'Times New Roman');
    end
    colormap(gca, cmapGlobal); cb3 = colorbar; cb3.Color = 'w'; cb3.Label.String = 'Pressure |P|'; caxis(climP);
    TpIMFunctions.apply3DViewStyle();
end

function renderFluidCavityShells(cavitiesStruct, faceAlpha, faceColor, edgeColor)
    for i = 1:length(cavitiesStruct)
        if ~isfield(cavitiesStruct, 'FluidVerts') || isempty(cavitiesStruct(i).FluidVerts) || isempty(cavitiesStruct(i).FluidFaces)
            continue;
        end
        verts = cavitiesStruct(i).FluidVerts;
        faces = cavitiesStruct(i).FluidFaces;
        for ff = 1:numel(faces)
            ids = faces{ff};
            if numel(ids) < 3, continue; end
            patch('Vertices', verts, 'Faces', ids, 'FaceColor', faceColor, 'FaceAlpha', faceAlpha, ...
                'EdgeColor', edgeColor, 'LineWidth', 0.25);
        end
    end
end

function renderFluidCavitiesByValue(cavitiesStruct, values, cmap, clim, faceAlpha, edgeColor)
    for i = 1:length(cavitiesStruct)
        if ~isfield(cavitiesStruct, 'FluidVerts') || isempty(cavitiesStruct(i).FluidVerts) || isempty(cavitiesStruct(i).FluidFaces)
            continue;
        end
        val = values(min(i, numel(values)));
        colorIdx = TpIMFunctions.mapValueToColorIndex(val, clim, size(cmap, 1));
        fc = cmap(colorIdx, :);
        verts = cavitiesStruct(i).FluidVerts;
        faces = cavitiesStruct(i).FluidFaces;
        for ff = 1:numel(faces)
            ids = faces{ff};
            if numel(ids) < 3, continue; end
            patch('Vertices', verts, 'Faces', ids, 'FaceColor', fc, 'FaceAlpha', faceAlpha, ...
                'EdgeColor', edgeColor, 'LineWidth', 0.20);
        end
    end
end

function renderActiveInterfaceFaces(polygonsStruct, V, edgesIdx, activeMask, values, cmap, clim, faceAlpha)
    activeIds = find(activeMask(:).');
    for kk = activeIds
        polyId = edgesIdx(kk, 3);
        if polyId < 1 || polyId > length(polygonsStruct), continue; end
        poly = polygonsStruct(polyId);
        if numel(poly.NodeIDs) < 3, continue; end
        colorIdx = TpIMFunctions.mapValueToColorIndex(values(kk), clim, size(cmap, 1));
        patch('Vertices', V, 'Faces', poly.NodeIDs, 'FaceColor', cmap(colorIdx, :), 'FaceAlpha', faceAlpha, ...
            'EdgeColor', [0.02 0.02 0.02], 'LineWidth', 0.40);
    end
end

function renderBranchLinesByValue(finalNodes, edgesIdx, activeMask, values, cmap, clim, lineWidth)
    activeIds = find(activeMask(:).');
    for kk = activeIds
        na = edgesIdx(kk, 1); nb = edgesIdx(kk, 2);
        colorIdx = TpIMFunctions.mapValueToColorIndex(values(kk), clim, size(cmap, 1));
        ptA = finalNodes(na, :); ptB = finalNodes(nb, :);
        plot3([ptA(1), ptB(1)], [ptA(2), ptB(2)], [ptA(3), ptB(3)], '-', ...
            'Color', cmap(colorIdx, :), 'LineWidth', lineWidth);
    end
end

function clim = calculateSafeColorLimits(values)
    vals = real(values(:));
    vals = vals(isfinite(vals));
    if isempty(vals)
        clim = [0, 1];
        return;
    end
    vmin = min(vals); vmax = max(vals);
    if abs(vmax - vmin) < 1e-12
        pad = max(abs(vmax), 1) * 0.05;
        clim = [vmin - pad, vmax + pad];
    else
        clim = [vmin, vmax];
    end
end

function idx = mapValueToColorIndex(value, clim, nColor)
    if ~isfinite(value)
        value = clim(1);
    end
    t = (real(value) - clim(1)) / max(clim(2) - clim(1), 1e-12);
    t = max(min(t, 1), 0);
    idx = round(t * (nColor - 1)) + 1;
end

function apply3DViewStyle()
    axis vis3d;
    camlight headlight;
    lighting gouraud;
    material dull;
    set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', 'FontName', 'Times New Roman');
end

function results = runMethodComparisonCore(cfg, x0, lb, ub, W, numTotalEdges, hardFaceGroups, cavityMembership, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray, bandRanges)
    % Run the fair optimizer-ablation methods under a matched TpIM budget.
    if ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    faceMembership = TpIMFunctions.buildFaceMembership(cavityMembership, edgesIdx, optIndices, W);
    nMethods = numel(cfg.methodCodes);
    template = struct('methodCode', '', 'methodName', '', 'methodIndex', [], 'run', [], ...
        'x', [], 'finalObjective', [], 'meanAlphaGlobal', [], 'meanAlphaShow', [], ...
        'traceEval', [], 'traceOptimizerBestf', [], 'traceGlobalBestf', [], ...
        'alphaGlobal', [], 'alphaShow', [], 'exitflags', [], 'outputs', [], ...
        'totalTpimEvals', [], 'stageLog', []);
    results = repmat(template, cfg.nRuns * nMethods, 1);
    row = 0;
    for runId = 1:cfg.nRuns
        for methodIndex = 1:nMethods
            methodCode = cfg.methodCodes{methodIndex};
            methodName = cfg.methodNames{methodIndex};
            seedValue = cfg.baseSeed + runId * 100 + methodIndex;
            rng(seedValue, 'twister');
            fprintf('\n=> [Optimizer comparison] Run %02d/%02d, %s\n', runId, cfg.nRuns, methodName);
            methodResult = TpIMFunctions.runMethodOnce(methodCode, cfg, x0, lb, ub, W, hardFaceGroups, faceMembership, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray, bandRanges);

            [nSideArr, rArr, bArr] = TpIMFunctions.decodeGenes(methodResult.x, W, numTotalEdges, optIndices, lSafeArray);
            finalObjective = TpIMFunctions.calculateGlobalFitness(methodResult.x, W, cfg.fGlobal, cfg.omegaGlobal, ...
                numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, ...
                tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray);
            [alphaFwd, alphaBwd, ~, ~] = TpIMFunctions.solveAcoustics( ...
                nSideArr, rArr, bArr, cfg.fGlobal, cfg.omegaGlobal, ...
                numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, ...
                tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot);
            [alphaShowFwd, alphaShowBwd, ~, ~] = TpIMFunctions.solveAcoustics( ...
                nSideArr, rArr, bArr, cfg.fShow, cfg.omegaShow, ...
                numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, ...
                tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot);
            if isBidirectional
                alphaGlobal = (alphaFwd + alphaBwd) / 2;
                alphaShow = (alphaShowFwd + alphaShowBwd) / 2;
            else
                alphaGlobal = alphaFwd;
                alphaShow = alphaShowFwd;
            end

            row = row + 1;
            results(row).methodCode = methodCode;
            results(row).methodName = methodName;
            results(row).methodIndex = methodIndex;
            results(row).run = runId;
            results(row).x = methodResult.x;
            results(row).finalObjective = finalObjective;
            results(row).meanAlphaGlobal = -finalObjective;
            results(row).meanAlphaShow = mean(alphaShow);
            results(row).traceEval = methodResult.traceEval;
            results(row).traceOptimizerBestf = methodResult.traceOptimizerBestf;
            results(row).traceGlobalBestf = methodResult.traceGlobalBestf;
            results(row).alphaGlobal = alphaGlobal;
            results(row).alphaShow = alphaShow;
            results(row).exitflags = methodResult.exitflags;
            results(row).outputs = methodResult.outputs;
            results(row).totalTpimEvals = methodResult.totalTpimEvals;
            results(row).stageLog = methodResult.stageLog;
            fprintf('   final J = %.6f, mean alpha = %.4f, TpIM evals = %d\n', ...
                results(row).finalObjective, results(row).meanAlphaGlobal, results(row).totalTpimEvals);
        end
        save(fullfile(cfg.outputDir, sprintf('OptimizerComparisonPartialAfterRun_%02d.mat', runId)), 'results', 'cfg', '-v7.3');
    end
end

function methodResult = runMethodOnce(methodCode, cfg, x0, lb, ub, W, hardFaceGroups, faceMembership, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray, bandRanges)
    nvars = numel(x0);
    intconFull = (2*W+1):nvars;
    xCurrent = x0;
    traceEval = [];
    traceOptimizerBestf = [];
    traceGlobalBestf = [];
    exitflags = [];
    outputs = {};
    stageLog = {};
    evalOffset = 0;

    switch upper(methodCode)
        case 'M1'
            stages = struct('name', {'global'}, 'group', {0}, 'weights', {[]}, 'type', {'global'}, 'activeFaces', {1:W});
        case 'M2'
            stages = TpIMFunctions.createFrequencyStages({1:W, 1:W, 1:W}, cfg);
        case 'M3'
            hardSets = {find(hardFaceGroups == 3)', find(hardFaceGroups == 2)', find(hardFaceGroups == 1)'};
            stages = TpIMFunctions.createFrequencyStages(hardSets, cfg);
        case 'M4'
            stages = TpIMFunctions.createFuzzyIndependentStages(faceMembership, hardFaceGroups, cfg);
        case 'M5'
            fuzzySets = TpIMFunctions.createFuzzyActiveSets(faceMembership, hardFaceGroups, cfg.overlapThreshold);
            stages = TpIMFunctions.createFrequencyStages(fuzzySets, cfg);
        case 'M6'
            fuzzySets = TpIMFunctions.createFuzzyActiveSets(faceMembership, hardFaceGroups, cfg.overlapThreshold);
            stages = TpIMFunctions.createFrequencyStages(fuzzySets, cfg);
            stages(end+1) = struct('name', 'global fine-tuning', 'group', 0, 'weights', [], 'type', 'fine', 'activeFaces', 1:W);
        otherwise
            error('Unknown method code: %s', methodCode);
    end

    independentStageSolutions = cell(1, 3);
    independentStageActive = false(W, 3);

    for stageId = 1:numel(stages)
        stage = stages(stageId);
        activeFaces = stage.activeFaces;
        if isempty(activeFaces)
            continue;
        end

        remainingBudget = cfg.totalEvalBudget - evalOffset;
        if remainingBudget < 2 * cfg.populationSize
            warning('[Optimizer comparison budget] Method %s stopped before stage "%s": remaining budget (%d) is smaller than one GA generation.', ...
                methodCode, stage.name, remainingBudget);
            break;
        end
        remainingStages = stages(stageId:end);
        [remainingGenerationPlan, remainingPlannedBudget, remainingBudgetWeights] = ...
            TpIMFunctions.planStageGenerations(remainingBudget, cfg.populationSize, remainingStages, W, cfg);
        stageGenerations = remainingGenerationPlan(1);
        plannedStageBudget = remainingPlannedBudget(1);
        stageBudgetWeight = remainingBudgetWeights(1);

        subIdx = [activeFaces, W + activeFaces, 2*W + activeFaces];
        subIdx = unique(subIdx, 'stable');
        xBaseForStage = xCurrent;
        if strcmpi(methodCode, 'M4')
            xBaseForStage = x0;
        end

        if strcmp(stage.type, 'global') || strcmp(stage.type, 'fine')
            objective = @(x) TpIMFunctions.calculateGlobalFitness(x, W, cfg.fGlobal, cfg.omegaGlobal, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray);
            localLb = lb;
            localUb = ub;
            localIntcon = intconFull;
            initial = xCurrent;
            diagFun = objective;
            if strcmp(stage.type, 'fine')
                localLb = max(lb, xCurrent * 0.5);
                localUb = min(ub, xCurrent * 1.5);
                localLb(2*W+1:end) = floor(localLb(2*W+1:end));
                localUb(2*W+1:end) = ceil(localUb(2*W+1:end));
                initial = min(max(xCurrent, localLb), localUb);
            end
        else
            weights = stage.weights;
            objective = @(xSub) TpIMFunctions.calculateWeightedFitness(xSub, subIdx, xBaseForStage, W, cfg.fGlobal, cfg.omegaGlobal, bandRanges, weights(1), weights(2), weights(3), numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray);
            localLb = lb(subIdx);
            localUb = ub(subIdx);
            localIntcon = (2*numel(activeFaces)+1):numel(subIdx);
            initial = xBaseForStage(subIdx);
            diagFun = @(xSub) TpIMFunctions.calculateGlobalFitness(TpIMFunctions.expandGeneSubvector(xSub, subIdx, xBaseForStage), W, cfg.fGlobal, cfg.omegaGlobal, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray);
        end
        initial = min(max(initial, localLb), localUb);

        evalOffsetBeforeStage = evalOffset;
        stageCfg = cfg;
        forceFullBudgetStage = strcmpi(methodCode, 'M1') || ...
            (strcmpi(methodCode, 'M6') && stageId == numel(stages));
        if forceFullBudgetStage && isfield(stageCfg, 'stageSwitch')
            stageCfg.stageSwitch.enabled = false;
        end
        [xStage, fvalStage, exitflagStage, outputStage, stageTrace] = TpIMFunctions.runGaStage(objective, numel(initial), localLb, localUb, localIntcon, initial, stageGenerations, stageCfg, evalOffset, diagFun);
        evalOffset = stageTrace.evalOffsetAfter;
        actualStageBudget = evalOffset - evalOffsetBeforeStage;
        traceEval = [traceEval; stageTrace.eval(:)]; 
        traceOptimizerBestf = [traceOptimizerBestf; stageTrace.optimizerBestf(:)]; 
        traceGlobalBestf = [traceGlobalBestf; stageTrace.globalBestf(:)]; 
        exitflags(end+1) = exitflagStage; 
        outputs{end+1} = outputStage; 
        stageLog{end+1} = struct('stage', stage.name, ...
            'generations', stageGenerations, ...
            'populationSize', cfg.populationSize, ...
            'plannedTpimBudget', plannedStageBudget, ...
            'actualTpimBudget', actualStageBudget, ...
            'remainingTpimBudgetBeforeStage', remainingBudget, ...
            'budgetWeight', stageBudgetWeight, ...
            'forceFullBudgetStage', forceFullBudgetStage, ...
            'stallGenerations', TpIMFunctions.getGaStallGenerations(stageCfg, stageGenerations), ...
            'functionTolerance', TpIMFunctions.getGaFunctionTolerance(stageCfg), ...
            'nvars', numel(initial), ...
            'activeFaces', activeFaces, ...
            'fval', fvalStage, ...
            'exitflag', exitflagStage, ...
            'evalOffsetAfter', evalOffset); 

        if strcmpi(methodCode, 'M4') && ~strcmp(stage.type, 'global') && ~strcmp(stage.type, 'fine')
            xStageFull = x0;
            xStageFull(subIdx) = xStage;
            independentStageSolutions{stage.group} = xStageFull;
            independentStageActive(activeFaces, stage.group) = true;
        elseif strcmp(stage.type, 'global') || strcmp(stage.type, 'fine')
            xCurrent = xStage;
        else
            xCurrent(subIdx) = xStage;
        end
    end

    if strcmpi(methodCode, 'M4')
        xCurrent = TpIMFunctions.mergeFuzzySolutions(x0, independentStageSolutions, independentStageActive, faceMembership, lb, ub, W);
    end
    xCurrent(2*W+1:end) = round(xCurrent(2*W+1:end));
    xCurrent = min(max(xCurrent, lb), ub);
    finalGlobal = TpIMFunctions.calculateGlobalFitness(xCurrent, W, cfg.fGlobal, cfg.omegaGlobal, numNodes, finalVolsM3, edgesIdx, lGeomM, sFaceM2, edgeType, tWallM, tTopM, rho0, c, eta, isBidirectional, sTotalM2, idxIncTop, idxIncBot, optIndices, lSafeArray);
    traceEval = [traceEval; evalOffset + 1];
    traceOptimizerBestf = [traceOptimizerBestf; finalGlobal];
    if isempty(traceGlobalBestf)
        traceGlobalBestf = finalGlobal;
    else
        traceGlobalBestf = [traceGlobalBestf; min(traceGlobalBestf(end), finalGlobal)];
    end
    traceGlobalBestf = cummin(traceGlobalBestf);

    methodResult = struct();
    methodResult.x = xCurrent;
    methodResult.traceEval = traceEval;
    methodResult.traceOptimizerBestf = traceOptimizerBestf;
    methodResult.traceGlobalBestf = traceGlobalBestf;
    methodResult.exitflags = exitflags;
    methodResult.outputs = outputs;
    methodResult.totalTpimEvals = max(traceEval);
    methodResult.stageLog = stageLog;
end

function stages = createFrequencyStages(activeSets, cfg)
    names = {'high-frequency stage', 'mid-frequency stage', 'low-frequency stage'};
    groups = {3, 2, 1};
    weights = {cfg.stageWeights.high, cfg.stageWeights.mid, cfg.stageWeights.low};
    stages = repmat(struct('name', '', 'group', 0, 'weights', [], 'type', 'weighted', 'activeFaces', []), 1, 3);
    for i = 1:3
        stages(i).name = names{i};
        stages(i).group = groups{i};
        stages(i).weights = weights{i};
        stages(i).type = 'weighted';
        stages(i).activeFaces = activeSets{i};
    end
end

function stages = createFuzzyIndependentStages(faceMembership, hardFaceGroups, cfg)
    activeSets = TpIMFunctions.createFuzzyActiveSets(faceMembership, hardFaceGroups, cfg.overlapThreshold);
    stages = TpIMFunctions.createFrequencyStages(activeSets, cfg);
    for i = 1:numel(stages)
        stages(i).name = ['independent ', stages(i).name];
    end
end

function activeSets = createFuzzyActiveSets(faceMembership, hardFaceGroups, threshold)
    activeSets = cell(1, 3);
    activeSets{1} = find(faceMembership(:,3)' >= threshold | hardFaceGroups' == 3);
    activeSets{2} = find(faceMembership(:,2)' >= threshold | hardFaceGroups' == 2);
    activeSets{3} = find(faceMembership(:,1)' >= threshold | hardFaceGroups' == 1);
end

function [generationPlan, plannedBudget, stageBudgetWeights] = planStageGenerations(totalEvalBudget, populationSize, stages, W, cfg)
    nStages = numel(stages);
    if nStages < 1
        generationPlan = [];
        plannedBudget = [];
        stageBudgetWeights = [];
        return;
    end
    activeStageMask = arrayfun(@(stage) ~isempty(stage.activeFaces), stages);
    if ~any(activeStageMask)
        generationPlan = zeros(1, nStages);
        plannedBudget = zeros(1, nStages);
        stageBudgetWeights = zeros(1, nStages);
        return;
    end

    if isfield(cfg, 'stageBudget') && isfield(cfg.stageBudget, 'minGenerations')
        minGenerations = max(1, round(cfg.stageBudget.minGenerations));
    else
        minGenerations = 1;
    end

    stageBudgetWeights = TpIMFunctions.computeStageBudgetWeights(stages, W, cfg);
    stageBudgetWeights(~activeStageMask) = 0;
    generationPlan = zeros(1, nStages);
    generationPlan(activeStageMask) = minGenerations;

    minEvalBudget = populationSize * sum(generationPlan(activeStageMask) + 1);
    remaining = totalEvalBudget - minEvalBudget;
    if remaining < 0
        warning('TpIM:BudgetTooSmall', ...
            'The requested totalEvalBudget is smaller than the minimum stage budget; using minimum generations for all stages.');
        plannedBudget = zeros(1, nStages);
        plannedBudget(activeStageMask) = populationSize * (generationPlan(activeStageMask) + 1);
        return;
    end

    extraBudget = remaining * stageBudgetWeights / sum(stageBudgetWeights);
    extraGenerations = floor(extraBudget / populationSize);
    generationPlan = generationPlan + extraGenerations;
    plannedBudget = zeros(1, nStages);
    plannedBudget(activeStageMask) = populationSize * (generationPlan(activeStageMask) + 1);

    remaining = totalEvalBudget - sum(plannedBudget);
    activeStageIds = find(activeStageMask);
    [~, localPriorityOrder] = sort(stageBudgetWeights(activeStageMask), 'descend');
    priorityOrder = activeStageIds(localPriorityOrder);
    orderId = 1;
    while remaining >= populationSize
        stageId = priorityOrder(orderId);
        generationPlan(stageId) = generationPlan(stageId) + 1;
        plannedBudget(stageId) = plannedBudget(stageId) + populationSize;
        remaining = remaining - populationSize;
        orderId = orderId + 1;
        if orderId > numel(priorityOrder)
            orderId = 1;
        end
    end
end

function stageBudgetWeights = computeStageBudgetWeights(stages, W, cfg)
    nStages = numel(stages);
    if ~isfield(cfg, 'stageBudget') || ~isfield(cfg.stageBudget, 'mode') || strcmpi(cfg.stageBudget.mode, 'uniform')
        stageBudgetWeights = ones(1, nStages);
        return;
    end

    if isfield(cfg.stageBudget, 'variableExponent')
        variableExponent = cfg.stageBudget.variableExponent;
    else
        variableExponent = 0.65;
    end
    if isfield(cfg.stageBudget, 'bandwidthExponent')
        bandwidthExponent = cfg.stageBudget.bandwidthExponent;
    else
        bandwidthExponent = 0.35;
    end

    stageBudgetWeights = ones(1, nStages);
    for stageId = 1:nStages
        stage = stages(stageId);
        activeRatio = max(numel(stage.activeFaces), 1) / max(W, 1);
        variableWeight = activeRatio ^ variableExponent;
        bandwidthWeight = TpIMFunctions.calculateStageBandwidthFraction(stage, cfg) ^ bandwidthExponent;
        typeWeight = TpIMFunctions.getStageTypeBudgetWeight(stage, cfg);
        stageBudgetWeights(stageId) = max(variableWeight * bandwidthWeight * typeWeight, eps);
    end
end

function bandwidthFraction = calculateStageBandwidthFraction(stage, cfg)
    if ~isfield(cfg, 'bandRanges') || ~isfield(cfg.bandRanges, 'global')
        bandwidthFraction = 1;
        return;
    end
    globalWidth = max(diff(cfg.bandRanges.global), eps);
    switch stage.group
        case 3
            if isfield(cfg.bandRanges, 'high')
                stageWidth = diff(cfg.bandRanges.high);
            else
                stageWidth = globalWidth;
            end
        case 2
            if isfield(cfg.bandRanges, 'mid')
                stageWidth = diff(cfg.bandRanges.mid);
            else
                stageWidth = globalWidth;
            end
        case 1
            if isfield(cfg.bandRanges, 'low')
                stageWidth = diff(cfg.bandRanges.low);
            else
                stageWidth = globalWidth;
            end
        otherwise
            stageWidth = globalWidth;
    end
    bandwidthFraction = max(stageWidth / globalWidth, eps);
end

function typeWeight = getStageTypeBudgetWeight(stage, cfg)
    typeWeight = 1;
    if ~isfield(cfg, 'stageBudget') || ~isfield(cfg.stageBudget, 'typeWeights')
        return;
    end
    typeWeights = cfg.stageBudget.typeWeights;
    if strcmp(stage.type, 'fine') && isfield(typeWeights, 'fine')
        typeWeight = typeWeights.fine;
    elseif strcmp(stage.type, 'global') && isfield(typeWeights, 'global')
        typeWeight = typeWeights.global;
    elseif stage.group == 3 && isfield(typeWeights, 'high')
        typeWeight = typeWeights.high;
    elseif stage.group == 2 && isfield(typeWeights, 'mid')
        typeWeight = typeWeights.mid;
    elseif stage.group == 1 && isfield(typeWeights, 'low')
        typeWeight = typeWeights.low;
    end
end

function [xOpt, fval, exitflag, output, stageTrace] = runGaStage(objective, nvars, lb, ub, intcon, initial, maxGenerations, cfg, evalOffset, diagFun)
    tpimGaTrace = struct();
    tpimGaTrace.baseEval = evalOffset;
    tpimGaTrace.eval = [];
    tpimGaTrace.optimizerBestf = [];
    tpimGaTrace.globalBestf = [];
    tpimGaTrace.globalBestSoFar = inf;
    tpimGaTrace.diagCount = 0;
    tpimGaTrace.diagFun = [];
    if isfield(cfg, 'diagnosticGlobalObjective') && cfg.diagnosticGlobalObjective
        tpimGaTrace.diagFun = diagFun;
    end
    TpIMFunctions.manageGaTrace('set', tpimGaTrace);
    traceCleanup = onCleanup(@() TpIMFunctions.manageGaTrace('clear')); %#ok<NASGU>

    options = TpIMFunctions.createGaOptions(cfg.populationSize, maxGenerations, cfg.useParallel, initial, cfg);
    [xOpt, fval, exitflag, output] = ga(objective, nvars, [], [], [], [], lb, ub, [], intcon, options);
    tpimGaTrace = TpIMFunctions.manageGaTrace('get');
    optimizerEvals = 0;
    if isfield(output, 'funccount')
        optimizerEvals = output.funccount;
    elseif ~isempty(tpimGaTrace.eval)
        optimizerEvals = max(tpimGaTrace.eval) - evalOffset - tpimGaTrace.diagCount;
    end
    evalOffsetAfter = evalOffset + optimizerEvals + tpimGaTrace.diagCount;

    if isempty(tpimGaTrace.eval)
        tpimGaTrace.eval = evalOffsetAfter;
        tpimGaTrace.optimizerBestf = fval;
        tpimGaTrace.globalBestf = fval;
    end
    if tpimGaTrace.eval(end) < evalOffsetAfter
        finalDiag = fval;
        if ~isempty(tpimGaTrace.diagFun)
            try
                finalDiag = tpimGaTrace.diagFun(xOpt);
                tpimGaTrace.diagCount = tpimGaTrace.diagCount + 1;
                evalOffsetAfter = evalOffsetAfter + 1;
            catch
                finalDiag = fval;
            end
        end
        tpimGaTrace.globalBestSoFar = min(tpimGaTrace.globalBestSoFar, finalDiag);
        tpimGaTrace.eval(end+1,1) = evalOffsetAfter;
        tpimGaTrace.optimizerBestf(end+1,1) = fval;
        tpimGaTrace.globalBestf(end+1,1) = tpimGaTrace.globalBestSoFar;
    end

    stageTrace = struct();
    stageTrace.eval = tpimGaTrace.eval;
    stageTrace.optimizerBestf = tpimGaTrace.optimizerBestf;
    stageTrace.globalBestf = tpimGaTrace.globalBestf;
    stageTrace.evalOffsetAfter = evalOffsetAfter;
    TpIMFunctions.manageGaTrace('clear');
end

function options = createGaOptions(populationSize, maxGenerations, useParallel, initial, cfg)
    if nargin < 5
        cfg = struct();
    end
    maxGenerations = max(1, round(maxGenerations));
    stallGenerations = TpIMFunctions.getGaStallGenerations(cfg, maxGenerations);
    functionTolerance = TpIMFunctions.getGaFunctionTolerance(cfg);
    args = {'PopulationSize', populationSize, ...
        'MaxGenerations', maxGenerations, ...
        'MaxStallGenerations', stallGenerations, ...
        'FunctionTolerance', functionTolerance, ...
        'Display', 'off', ...
        'UseParallel', useParallel, ...
        'OutputFcn', @TpIMFunctions.recordGaTrace};
    if ~isempty(initial)
        args = [args, {'InitialPopulationMatrix', initial}];
    end
    options = optimoptions('ga', args{:});
end

function stallGenerations = getGaStallGenerations(cfg, maxGenerations)
    maxGenerations = max(1, round(maxGenerations));
    adaptiveSwitch = isfield(cfg, 'stageSwitch') && isfield(cfg.stageSwitch, 'enabled') && cfg.stageSwitch.enabled;
    if adaptiveSwitch
        if isfield(cfg.stageSwitch, 'stallGenerations')
            requestedStall = cfg.stageSwitch.stallGenerations;
        else
            requestedStall = max(4, round(0.08 * maxGenerations));
        end
        stallGenerations = max(1, min(maxGenerations, round(requestedStall)));
    else
        stallGenerations = max(maxGenerations + 1, 50);
    end
end

function functionTolerance = getGaFunctionTolerance(cfg)
    adaptiveSwitch = isfield(cfg, 'stageSwitch') && isfield(cfg.stageSwitch, 'enabled') && cfg.stageSwitch.enabled;
    if adaptiveSwitch
        if isfield(cfg.stageSwitch, 'functionTolerance')
            functionTolerance = cfg.stageSwitch.functionTolerance;
        else
            functionTolerance = 1e-5;
        end
    else
        functionTolerance = 1e-12;
    end
end

function [state, options, optchanged] = recordGaTrace(options, state, flag)
    optchanged = false;
    tpimGaTrace = TpIMFunctions.manageGaTrace('get');
    if isempty(tpimGaTrace) || ~(strcmp(flag, 'init') || strcmp(flag, 'iter') || strcmp(flag, 'done'))
        return;
    end
    if isempty(state.Score)
        return;
    end
    optimizerBest = min(state.Score);
    if isfield(state, 'Best') && ~isempty(state.Best)
        optimizerBest = state.Best(end);
    end
    diagnosticBest = optimizerBest;
    if isfield(tpimGaTrace, 'diagFun') && ~isempty(tpimGaTrace.diagFun) && ~isempty(state.Population)
        try
            [~, idxBest] = min(state.Score);
            diagnosticBest = tpimGaTrace.diagFun(state.Population(idxBest, :));
            tpimGaTrace.diagCount = tpimGaTrace.diagCount + 1;
        catch
            diagnosticBest = optimizerBest;
        end
    end
    tpimGaTrace.globalBestSoFar = min(tpimGaTrace.globalBestSoFar, diagnosticBest);
    evalCount = tpimGaTrace.baseEval + state.FunEval + tpimGaTrace.diagCount;
    if ~isempty(tpimGaTrace.eval) && tpimGaTrace.eval(end) == evalCount
        tpimGaTrace.optimizerBestf(end) = optimizerBest;
        tpimGaTrace.globalBestf(end) = tpimGaTrace.globalBestSoFar;
    else
        tpimGaTrace.eval(end+1,1) = evalCount;
        tpimGaTrace.optimizerBestf(end+1,1) = optimizerBest;
        tpimGaTrace.globalBestf(end+1,1) = tpimGaTrace.globalBestSoFar;
    end
    TpIMFunctions.manageGaTrace('set', tpimGaTrace);
end

function traceState = manageGaTrace(action, value)
    % Store one GA trace without exposing mutable global state.
    persistent storedTrace;
    if nargin < 2
        value = [];
    end
    switch lower(action)
        case 'set'
            storedTrace = value;
        case 'get'
            % Return the current state below.
        case 'clear'
            storedTrace = [];
        otherwise
            error('TpIMFunctions:UnknownTraceAction', ...
                'Unknown GA trace action: %s.', action);
    end
    traceState = storedTrace;
end

function xFull = expandGeneSubvector(xSub, subIndices, xBase)
    xFull = xBase;
    xFull(subIndices) = xSub;
end

function xMerged = mergeFuzzySolutions(x0, stageSolutions, stageActive, faceMembership, lb, ub, W)
    xMerged = x0;
    for faceId = 1:W
        genePos = [faceId, W + faceId, 2*W + faceId];
        weights = faceMembership(faceId, :) .* stageActive(faceId, :);
        if sum(weights) <= 0
            [~, owner] = max(faceMembership(faceId, :));
            weights(owner) = 1;
        end
        weights = weights / sum(weights);
        for localGene = 1:3
            value = 0;
            for groupId = 1:3
                if isempty(stageSolutions{groupId})
                    candidate = x0(genePos(localGene));
                else
                    candidate = stageSolutions{groupId}(genePos(localGene));
                end
                value = value + weights(groupId) * candidate;
            end
            xMerged(genePos(localGene)) = value;
        end
    end
    xMerged(2*W+1:end) = round(xMerged(2*W+1:end));
    xMerged = min(max(xMerged, lb), ub);
end

function faceMembership = buildFaceMembership(cavityMembership, edgesIdx, optIndices, W)
    nGroups = size(cavityMembership, 2);
    faceMembership = zeros(W, nGroups);
    numCavs = size(cavityMembership, 1);
    for i = 1:W
        c1 = edgesIdx(optIndices(i), 1) - 1;
        c2 = edgesIdx(optIndices(i), 2) - 1;
        rows = [];
        if c1 > 0 && c1 <= numCavs
            rows = [rows; cavityMembership(c1, :)]; 
        end
        if c2 > 0 && c2 <= numCavs
            rows = [rows; cavityMembership(c2, :)]; 
        end
        if isempty(rows)
            faceMembership(i, :) = ones(1, nGroups) / nGroups;
        else
            faceMembership(i, :) = mean(rows, 1);
            s = sum(faceMembership(i, :));
            if s > 0
                faceMembership(i, :) = faceMembership(i, :) / s;
            else
                faceMembership(i, :) = ones(1, nGroups) / nGroups;
            end
        end
    end
end

function [figHandle, summaryTable] = renderMethodComparisonFigure(results, cfg)
    if ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    methodCodes = cellstr(string(cfg.methodCodes));
    methodNames = cellstr(string(cfg.methodNames));
    nMethods = numel(methodCodes);
    nRuns = max(cfg.nRuns, max([results.run]));
    if isfield(cfg, 'convergenceGridPoints')
        nGridPoints = cfg.convergenceGridPoints;
    else
        nGridPoints = 300;
    end
    if isfield(cfg, 'efficiencyAlphaTargets')
        alphaTargets = cfg.efficiencyAlphaTargets(:)';
    else
        alphaTargets = [0.70, 0.73, 0.75];
    end
    nTargets = numel(alphaTargets);

    colors = TpIMFunctions.getMethodColors(methodCodes);

    maxEval = max([results.totalTpimEvals]);
    evalGrid = linspace(1, maxEval, nGridPoints)';
    alphaCurves = nan(numel(evalGrid), nMethods, nRuns);
    medianCurves = nan(numel(evalGrid), nMethods);
    q1Curves = nan(numel(evalGrid), nMethods);
    q3Curves = nan(numel(evalGrid), nMethods);
    finalObjMatrix = nan(nRuns, nMethods);
    finalAlphaMatrix = nan(nRuns, nMethods);
    timeToTarget = nan(nRuns, nMethods, nTargets);
    smoothWindowPoints = TpIMFunctions.getConvergenceSmoothingWindow(nGridPoints, cfg);
    medianPlotCurves = nan(numel(evalGrid), nMethods);
    q1PlotCurves = nan(numel(evalGrid), nMethods);
    q3PlotCurves = nan(numel(evalGrid), nMethods);

    figHandle = figure('Name', 'M1-M6 optimizer ablation', ...
        'Color', 'w', 'Position', [100, 100, 1120, 840]);
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile(1);
    hold on; grid on; box on;
    for m = 1:nMethods
        idx = find(strcmp({results.methodCode}, methodCodes{m}));
        lightColor = colors(m, :) + (1 - colors(m, :)) * 0.78;
        for k = 1:numel(idx)
            r = results(idx(k));
            runId = r.run;
            alphaTrace = -r.traceGlobalBestf(:);
            plot(r.traceEval(:), alphaTrace, 'LineWidth', 0.45, 'Color', lightColor, 'HandleVisibility', 'off');
            switchEval = TpIMFunctions.getStageSwitchEvaluations(r.stageLog);
            if ~isempty(switchEval)
                switchAlpha = TpIMFunctions.resampleBestTrace(r.traceEval, alphaTrace, switchEval);
                scatter(switchEval, switchAlpha, 18, colors(m, :), 'filled', ...
                    'MarkerFaceAlpha', 0.72, 'MarkerEdgeColor', 'w', 'LineWidth', 0.35, ...
                    'HandleVisibility', 'off');
            end
            alphaCurves(:, m, runId) = TpIMFunctions.resampleBestTrace(r.traceEval, alphaTrace, evalGrid);
            finalObjMatrix(runId, m) = r.finalObjective;
            finalAlphaMatrix(runId, m) = r.meanAlphaGlobal;
            for t = 1:nTargets
                timeToTarget(runId, m, t) = TpIMFunctions.findFirstEvaluationAtTarget(r.traceEval, alphaTrace, alphaTargets(t));
            end
        end
    end
    legendHandles = gobjects(nMethods, 1);
    for m = 1:nMethods
        curvesM = reshape(alphaCurves(:, m, :), numel(evalGrid), nRuns);
        medianCurves(:, m) = TpIMFunctions.rowPercentileIgnoringNaN(curvesM, 50);
        q1Curves(:, m) = TpIMFunctions.rowPercentileIgnoringNaN(curvesM, 25);
        q3Curves(:, m) = TpIMFunctions.rowPercentileIgnoringNaN(curvesM, 75);
        medianPlotCurves(:, m) = TpIMFunctions.smoothConvergenceCurve(medianCurves(:, m), smoothWindowPoints);
        q1PlotCurves(:, m) = TpIMFunctions.smoothConvergenceCurve(q1Curves(:, m), smoothWindowPoints);
        q3PlotCurves(:, m) = TpIMFunctions.smoothConvergenceCurve(q3Curves(:, m), smoothWindowPoints);
        q1PlotCurves(:, m) = min(q1PlotCurves(:, m), medianPlotCurves(:, m));
        q3PlotCurves(:, m) = max(q3PlotCurves(:, m), medianPlotCurves(:, m));
        legendHandles(m) = plot(evalGrid, medianPlotCurves(:, m), 'LineWidth', 2.2, 'Color', colors(m, :));
    end
    xlabel('TpIM evaluation count', 'FontName', 'Times New Roman');
    ylabel('Best mean absorption', 'FontName', 'Times New Roman');
    title('Independent traces and stage-switch points', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    legend(legendHandles, methodCodes, 'Location', 'southeast', 'FontName', 'Times New Roman');
    ylim([0, 1]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    nexttile(2);
    hold on; grid on; box on;
    panelBHandles = gobjects(nMethods, 1);
    for m = 1:nMethods
        fill([evalGrid; flipud(evalGrid)], [q1PlotCurves(:, m); flipud(q3PlotCurves(:, m))], colors(m, :), ...
            'FaceAlpha', 0.14, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        panelBHandles(m) = plot(evalGrid, medianPlotCurves(:, m), 'LineWidth', 2.2, 'Color', colors(m, :));
    end
    xlabel('TpIM evaluation count', 'FontName', 'Times New Roman');
    ylabel('Median best mean absorption', 'FontName', 'Times New Roman');
    title('Smoothed median convergence with IQR band', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    legend(panelBHandles, methodCodes, 'Location', 'southeast', 'FontName', 'Times New Roman');
    ylim([0, 1]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    nexttile(3);
    hold on; grid on; box on;
    for m = 1:nMethods
        y = finalAlphaMatrix(:, m);
        x = m * ones(size(y));
        boxchart(x, y, 'BoxFaceColor', colors(m, :), 'BoxFaceAlpha', 0.45, 'MarkerStyle', 'none', 'BoxWidth', 0.45);
        jitter = (rand(size(y)) - 0.5) * 0.16;
        scatter(x + jitter, y, 18, colors(m, :), 'filled', 'MarkerFaceAlpha', 0.50);
    end
    xticks(1:nMethods);
    xticklabels(methodCodes);
    xlabel('Optimization method', 'FontName', 'Times New Roman');
    ylabel('Final best mean absorption', 'FontName', 'Times New Roman');
    title('Final-performance distribution', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    ylim([0, 1]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    nexttile(4);
    hold on; grid on; box on;
    if nMethods == 1
        offsets = 0;
        boxWidth = 0.12;
    else
        groupSpan = min(0.48, 0.22 + 0.05 * max(nMethods - 2, 0));
        offsets = linspace(-groupSpan / 2, groupSpan / 2, nMethods);
        boxWidth = min(0.10, max(0.06, 0.65 * groupSpan / max(nMethods - 1, 1)));
    end
    if isfield(cfg, 'totalEvalBudget')
        targetBudget = cfg.totalEvalBudget;
    else
        targetBudget = maxEval;
    end
    markFailedTargets = isfield(cfg, 'convergencePlot') && ...
        isfield(cfg.convergencePlot, 'markFailedTargets') && cfg.convergencePlot.markFailedTargets;
    finiteTargetValues = timeToTarget(isfinite(timeToTarget));
    if isempty(finiteTargetValues)
        yTop = targetBudget * 1.05;
    else
        yTop = max([targetBudget; finiteTargetValues(:)]) * 1.05;
    end
    if markFailedTargets
        yline(targetBudget, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, 'HandleVisibility', 'off');
    end
    for t = 1:nTargets
        for m = 1:nMethods
            yAll = timeToTarget(:, m, t);
            y = yAll(isfinite(yAll));
            nSuccess = numel(y);
            xCenter = t + offsets(m);
            if isempty(y)
                if markFailedTargets
                    plot(xCenter, targetBudget, 'x', 'Color', colors(m, :), ...
                        'MarkerSize', 8, 'LineWidth', 1.6, 'HandleVisibility', 'off');
                else
                    continue;
                end
            else
                x = xCenter * ones(size(y));
                boxchart(x, y, 'BoxFaceColor', colors(m, :), 'BoxFaceAlpha', 0.45, 'MarkerStyle', 'none', 'BoxWidth', boxWidth);
                jitter = (rand(size(y)) - 0.5) * boxWidth;
                scatter(x + jitter, y, 16, colors(m, :), 'filled', 'MarkerFaceAlpha', 0.45);
            end
            if markFailedTargets
                text(xCenter, yTop * 0.985, sprintf('%d/%d', nSuccess, nRuns), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                    'FontName', 'Times New Roman', 'FontSize', 8, 'Color', colors(m, :));
            end
        end
    end
    xticks(1:nTargets);
    targetLabels = cell(1, nTargets);
    for t = 1:nTargets
        targetLabels{t} = sprintf('\\alpha >= %.2f', alphaTargets(t));
    end
    xticklabels(targetLabels);
    xlabel('Target best mean absorption', 'FontName', 'Times New Roman');
    ylabel('TpIM evaluations to target', 'FontName', 'Times New Roman');
    title('Time-to-target efficiency', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    ylim([0, yTop]);
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    exportgraphics(figHandle, fullfile(cfg.outputDir, 'M1M6OptimizerAblation.png'), 'Resolution', 600);
    exportgraphics(figHandle, fullfile(cfg.outputDir, 'M1M6OptimizerAblation.pdf'), 'ContentType', 'vector');
    savefig(figHandle, fullfile(cfg.outputDir, 'M1M6OptimizerAblation.fig'));

    methodCol = strings(numel(results), 1);
    runCol = zeros(numel(results), 1);
    finalObjCol = zeros(numel(results), 1);
    meanAlphaCol = zeros(numel(results), 1);
    evalCol = zeros(numel(results), 1);
    targetEvalCols = nan(numel(results), nTargets);
    targetReachedCols = false(numel(results), nTargets);
    for i = 1:numel(results)
        methodCol(i) = string(results(i).methodCode);
        runCol(i) = results(i).run;
        finalObjCol(i) = results(i).finalObjective;
        meanAlphaCol(i) = results(i).meanAlphaGlobal;
        evalCol(i) = results(i).totalTpimEvals;
        alphaTrace = -results(i).traceGlobalBestf(:);
        for t = 1:nTargets
            evalToTarget = TpIMFunctions.findFirstEvaluationAtTarget(results(i).traceEval, alphaTrace, alphaTargets(t));
            targetEvalCols(i, t) = evalToTarget;
            targetReachedCols(i, t) = isfinite(evalToTarget);
        end
    end
    summaryTable = table(methodCol, runCol, finalObjCol, meanAlphaCol, evalCol, ...
        'VariableNames', {'Method', 'Run', 'FinalObjective', 'MeanAlphaGlobal', 'TpIMEvaluations'});
    for t = 1:nTargets
        targetId = sprintf('%d', t);
        summaryTable.(['AlphaTarget', targetId]) = repmat(alphaTargets(t), height(summaryTable), 1);
        summaryTable.(['EvalToTarget', targetId]) = targetEvalCols(:, t);
        summaryTable.(['ReachedTarget', targetId]) = targetReachedCols(:, t);
    end

    convTable = table(evalGrid, 'VariableNames', {'TpIMEvaluations'});
    allCurveTable = table(evalGrid, 'VariableNames', {'TpIMEvaluations'});
    for m = 1:nMethods
        convTable.([methodCodes{m}, '_MedianAlpha']) = medianCurves(:, m);
        convTable.([methodCodes{m}, '_Q1Alpha']) = q1Curves(:, m);
        convTable.([methodCodes{m}, '_Q3Alpha']) = q3Curves(:, m);
        for r = 1:nRuns
            allCurveTable.(sprintf('%sRun%02d_Alpha', methodCodes{m}, r)) = alphaCurves(:, m, r);
        end
    end
    writetable(convTable, fullfile(cfg.outputDir, 'M1M6ConvergenceGrid.csv'));
    writetable(allCurveTable, fullfile(cfg.outputDir, 'M1M6AllConvergenceCurves.csv'));
    writetable(summaryTable, fullfile(cfg.outputDir, 'M1M6Summary.csv'));

    targetRows = nMethods * nTargets;
    effMethod = strings(targetRows, 1);
    effTarget = zeros(targetRows, 1);
    effMedian = nan(targetRows, 1);
    effQ1 = nan(targetRows, 1);
    effQ3 = nan(targetRows, 1);
    effSuccess = nan(targetRows, 1);
    row = 0;
    for m = 1:nMethods
        for t = 1:nTargets
            row = row + 1;
            y = timeToTarget(:, m, t);
            effMethod(row) = string(methodCodes{m});
            effTarget(row) = alphaTargets(t);
            effMedian(row) = TpIMFunctions.vectorPercentileIgnoringNaN(y, 50);
            effQ1(row) = TpIMFunctions.vectorPercentileIgnoringNaN(y, 25);
            effQ3(row) = TpIMFunctions.vectorPercentileIgnoringNaN(y, 75);
            effSuccess(row) = mean(isfinite(y));
        end
    end
    efficiencyTable = table(effMethod, effTarget, effMedian, effQ1, effQ3, effSuccess, ...
        'VariableNames', {'Method', 'AlphaTarget', 'MedianEvalToTarget', 'Q1EvalToTarget', 'Q3EvalToTarget', 'SuccessRate'});
    writetable(efficiencyTable, fullfile(cfg.outputDir, 'M1M6EfficiencySummary.csv'));

    idxM1 = find(strcmp(methodCodes, 'M1'), 1);
    idxM6 = find(strcmp(methodCodes, 'M6'), 1);
    if ~isempty(idxM1) && ~isempty(idxM6)
        speedTarget = alphaTargets(:);
        medianM1 = nan(nTargets, 1);
        medianM6 = nan(nTargets, 1);
        speedupM1OverM6 = nan(nTargets, 1);
        successM1 = nan(nTargets, 1);
        successM6 = nan(nTargets, 1);
        for t = 1:nTargets
            yM1 = timeToTarget(:, idxM1, t);
            yM6 = timeToTarget(:, idxM6, t);
            medianM1(t) = TpIMFunctions.vectorPercentileIgnoringNaN(yM1, 50);
            medianM6(t) = TpIMFunctions.vectorPercentileIgnoringNaN(yM6, 50);
            speedupM1OverM6(t) = medianM1(t) / medianM6(t);
            successM1(t) = mean(isfinite(yM1));
            successM6(t) = mean(isfinite(yM6));
        end
        speedupTable = table(speedTarget, medianM1, medianM6, speedupM1OverM6, successM1, successM6, ...
            'VariableNames', {'AlphaTarget', 'M1MedianEval', 'M6MedianEval', 'M1OverM6Speedup', 'M1SuccessRate', 'M6SuccessRate'});
        writetable(speedupTable, fullfile(cfg.outputDir, 'M6VsM1Efficiency.csv'));
    end

    nameTable = table(string(methodCodes(:)), string(methodNames(:)), ...
        'VariableNames', {'Method', 'Definition'});
    writetable(nameTable, fullfile(cfg.outputDir, 'M1M6MethodDefinitions.csv'));
end

function [figHandle, summaryTable, speedupTable] = renderScalingStudyFigure(scalingTable, cfg)
    if ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    if isfield(cfg, 'efficiencyAlphaTargets')
        alphaTargets = cfg.efficiencyAlphaTargets(:)';
    else
        alphaTargets = [0.70, 0.73, 0.75];
    end
    if isfield(cfg, 'primaryAlphaTarget')
        primaryTarget = cfg.primaryAlphaTarget;
    else
        primaryTarget = alphaTargets(min(2, numel(alphaTargets)));
    end

    methodCodes = {'M1', 'M6'};
    colors = TpIMFunctions.getMethodColors(methodCodes);
    caseLabels = unique(string(scalingTable.CaseLabel), 'stable');
    nCases = numel(caseLabels);
    nTargets = numel(alphaTargets);
    primaryTargetIdx = TpIMFunctions.getPrimaryTargetIndex(alphaTargets, primaryTarget);
    primaryTarget = alphaTargets(primaryTargetIdx);

    nvarsCase = nan(nCases, 1);
    wCase = nan(nCases, 1);
    numCellsCase = nan(nCases, 1);
    medianEvalPrimary = nan(nCases, 2);
    q1EvalPrimary = nan(nCases, 2);
    q3EvalPrimary = nan(nCases, 2);
    successPrimary = nan(nCases, 2);
    medianFinalAlpha = nan(nCases, 2);
    q1FinalAlpha = nan(nCases, 2);
    q3FinalAlpha = nan(nCases, 2);
    speedupByTarget = nan(nCases, nTargets);

    summaryMethod = strings(nCases * numel(methodCodes) * nTargets, 1);
    summaryCase = strings(nCases * numel(methodCodes) * nTargets, 1);
    summaryNx = nan(numel(summaryCase), 1);
    summaryNy = nan(numel(summaryCase), 1);
    summaryNz = nan(numel(summaryCase), 1);
    summaryCells = nan(numel(summaryCase), 1);
    summaryW = nan(numel(summaryCase), 1);
    summaryNvars = nan(numel(summaryCase), 1);
    summaryTarget = nan(numel(summaryCase), 1);
    summaryMedianEval = nan(numel(summaryCase), 1);
    summaryQ1Eval = nan(numel(summaryCase), 1);
    summaryQ3Eval = nan(numel(summaryCase), 1);
    summarySuccess = nan(numel(summaryCase), 1);
    summaryMedianAlpha = nan(numel(summaryCase), 1);
    row = 0;

    for cId = 1:nCases
        caseIdx = string(scalingTable.CaseLabel) == caseLabels(cId);
        firstCaseRow = find(caseIdx, 1, 'first');
        nvarsCase(cId) = scalingTable.Nvars(firstCaseRow);
        wCase(cId) = scalingTable.W(firstCaseRow);
        numCellsCase(cId) = scalingTable.NumCells(firstCaseRow);
        for m = 1:numel(methodCodes)
            methodIdx = caseIdx & strcmp(string(scalingTable.Method), methodCodes{m});
            finalAlpha = scalingTable.MeanAlphaGlobal(methodIdx);
            medianFinalAlpha(cId, m) = TpIMFunctions.vectorPercentileIgnoringNaN(finalAlpha, 50);
            q1FinalAlpha(cId, m) = TpIMFunctions.vectorPercentileIgnoringNaN(finalAlpha, 25);
            q3FinalAlpha(cId, m) = TpIMFunctions.vectorPercentileIgnoringNaN(finalAlpha, 75);

            primaryEval = TpIMFunctions.getScalingTargetEvaluation( ...
                scalingTable, methodIdx, primaryTargetIdx);
            medianEvalPrimary(cId, m) = TpIMFunctions.vectorPercentileIgnoringNaN(primaryEval, 50);
            q1EvalPrimary(cId, m) = TpIMFunctions.vectorPercentileIgnoringNaN(primaryEval, 25);
            q3EvalPrimary(cId, m) = TpIMFunctions.vectorPercentileIgnoringNaN(primaryEval, 75);
            successPrimary(cId, m) = mean(isfinite(primaryEval));

            for t = 1:nTargets
                targetEval = TpIMFunctions.getScalingTargetEvaluation( ...
                    scalingTable, methodIdx, t);
                row = row + 1;
                summaryCase(row) = caseLabels(cId);
                summaryMethod(row) = methodCodes{m};
                summaryNx(row) = scalingTable.Nx(firstCaseRow);
                summaryNy(row) = scalingTable.Ny(firstCaseRow);
                summaryNz(row) = scalingTable.Nz(firstCaseRow);
                summaryCells(row) = scalingTable.NumCells(firstCaseRow);
                summaryW(row) = scalingTable.W(firstCaseRow);
                summaryNvars(row) = scalingTable.Nvars(firstCaseRow);
                summaryTarget(row) = alphaTargets(t);
                summaryMedianEval(row) = TpIMFunctions.vectorPercentileIgnoringNaN(targetEval, 50);
                summaryQ1Eval(row) = TpIMFunctions.vectorPercentileIgnoringNaN(targetEval, 25);
                summaryQ3Eval(row) = TpIMFunctions.vectorPercentileIgnoringNaN(targetEval, 75);
                summarySuccess(row) = mean(isfinite(targetEval));
                summaryMedianAlpha(row) = medianFinalAlpha(cId, m);
            end
        end
        for t = 1:nTargets
            yM1 = TpIMFunctions.getScalingTargetEvaluation(scalingTable, ...
                caseIdx & strcmp(string(scalingTable.Method), 'M1'), t);
            yM6 = TpIMFunctions.getScalingTargetEvaluation(scalingTable, ...
                caseIdx & strcmp(string(scalingTable.Method), 'M6'), t);
            speedupByTarget(cId, t) = TpIMFunctions.vectorPercentileIgnoringNaN(yM1, 50) / ...
                TpIMFunctions.vectorPercentileIgnoringNaN(yM6, 50);
        end
    end

    summaryTable = table(summaryCase, summaryNx, summaryNy, summaryNz, summaryCells, summaryW, summaryNvars, ...
        summaryMethod, summaryTarget, summaryMedianEval, summaryQ1Eval, summaryQ3Eval, summarySuccess, summaryMedianAlpha, ...
        'VariableNames', {'CaseLabel', 'Nx', 'Ny', 'Nz', 'NumCells', 'W', 'Nvars', 'Method', 'AlphaTarget', ...
        'MedianEvalToTarget', 'Q1EvalToTarget', 'Q3EvalToTarget', 'SuccessRate', 'MedianFinalAlpha'});

    speedupCase = strings(nCases * nTargets, 1);
    speedupTarget = nan(nCases * nTargets, 1);
    speedupNvars = nan(nCases * nTargets, 1);
    speedupW = nan(nCases * nTargets, 1);
    speedupValue = nan(nCases * nTargets, 1);
    row = 0;
    for cId = 1:nCases
        for t = 1:nTargets
            row = row + 1;
            speedupCase(row) = caseLabels(cId);
            speedupTarget(row) = alphaTargets(t);
            speedupNvars(row) = nvarsCase(cId);
            speedupW(row) = wCase(cId);
            speedupValue(row) = speedupByTarget(cId, t);
        end
    end
    speedupTable = table(speedupCase, speedupTarget, speedupW, speedupNvars, speedupValue, ...
        'VariableNames', {'CaseLabel', 'AlphaTarget', 'W', 'Nvars', 'M1OverM6Speedup'});

    [nvarsPlot, plotOrder] = sort(nvarsCase);
    caseLabelsPlot = caseLabels(plotOrder);
    numCellsPlot = numCellsCase(plotOrder);
    medianEvalPlot = medianEvalPrimary(plotOrder, :);
    q1EvalPlot = q1EvalPrimary(plotOrder, :);
    q3EvalPlot = q3EvalPrimary(plotOrder, :);
    speedupPlot = speedupByTarget(plotOrder, :);
    successPlot = successPrimary(plotOrder, :);

    figHandle = figure('Name', 'M1-M6 scale-effect efficiency study', ...
        'Color', 'w', 'Position', [100, 100, 1680, 840]);
    tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile(1);
    hold on; grid on; box on;
    xCase = 1:nCases;
    bar(xCase, numCellsPlot, 0.62, 'FaceColor', [0.55 0.60 0.65], 'EdgeColor', [0.20 0.20 0.20]);
    for cId = 1:nCases
        text(xCase(cId), numCellsPlot(cId), sprintf('%d', numCellsPlot(cId)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontName', 'Times New Roman', 'FontSize', 10);
    end
    xticks(xCase);
    xticklabels(cellstr(caseLabelsPlot));
    xlabel('Unit-cell array', 'FontName', 'Times New Roman');
    ylabel('Number of unit cells', 'FontName', 'Times New Roman');
    title('Cell-array scale', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    nexttile(2, [1 2]);
    hold on; grid on; box on;
    for m = 1:numel(methodCodes)
        fill([nvarsPlot; flipud(nvarsPlot)], [q1EvalPlot(:, m); flipud(q3EvalPlot(:, m))], ...
            colors(m, :), 'FaceAlpha', 0.14, 'EdgeColor', 'none');
        plot(nvarsPlot, medianEvalPlot(:, m), '-o', 'LineWidth', 2.2, 'Color', colors(m, :), ...
            'MarkerFaceColor', colors(m, :), 'MarkerSize', 6);
    end
    xlabel('Number of design variables, 3W', 'FontName', 'Times New Roman');
    ylabel(sprintf('TpIM evaluations to \\alpha >= %.2f', primaryTarget), 'FontName', 'Times New Roman');
    title('Time-to-target scaling', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    legend(methodCodes, 'Location', 'northwest', 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    nexttile(4, [1 2]);
    hold on; grid on; box on;
    targetColors = lines(nTargets);
    for t = 1:nTargets
        plot(nvarsPlot, speedupPlot(:, t), '-o', 'LineWidth', 2.2, 'Color', targetColors(t, :), ...
            'MarkerFaceColor', targetColors(t, :), 'MarkerSize', 6);
    end
    yline(1, ':', 'Color', [0.35 0.35 0.35]);
    xlabel('Number of design variables, 3W', 'FontName', 'Times New Roman');
    ylabel('M1 median eval / M6 median eval', 'FontName', 'Times New Roman');
    title('M6 efficiency gain over M1', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    targetLegend = cell(1, nTargets);
    for t = 1:nTargets
        targetLegend{t} = sprintf('\\alpha >= %.2f', alphaTargets(t));
    end
    legend(targetLegend, 'Location', 'northwest', 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    nexttile(6);
    hold on; grid on; box on;
    for m = 1:numel(methodCodes)
        plot(nvarsPlot, successPlot(:, m), '-o', 'LineWidth', 2.0, 'Color', colors(m, :), ...
            'MarkerFaceColor', colors(m, :), 'MarkerSize', 6);
    end
    xlabel('Number of design variables, 3W', 'FontName', 'Times New Roman');
    ylabel(sprintf('Success rate for \\alpha >= %.2f', primaryTarget), 'FontName', 'Times New Roman');
    title('Target-reaching reliability', 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    ylim([0, 1.05]);
    legend(methodCodes, 'Location', 'southwest', 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

    exportgraphics(figHandle, fullfile(cfg.outputDir, 'M1M6ScaleEffect.png'), 'Resolution', 600);
    exportgraphics(figHandle, fullfile(cfg.outputDir, 'M1M6ScaleEffect.pdf'), 'ContentType', 'vector');
    savefig(figHandle, fullfile(cfg.outputDir, 'M1M6ScaleEffect.fig'));
    writetable(summaryTable, fullfile(cfg.outputDir, 'm1M6ScalingSummary.csv'));
    writetable(speedupTable, fullfile(cfg.outputDir, 'm1M6ScalingSpeedup.csv'));
end

function evalToTarget = findFirstEvaluationAtTarget(traceEval, traceAlpha, alphaTarget)
    traceEval = traceEval(:);
    traceAlpha = traceAlpha(:);
    idx = find(traceAlpha >= alphaTarget, 1, 'first');
    if isempty(idx)
        evalToTarget = NaN;
    else
        evalToTarget = traceEval(idx);
    end
end

function colors = getMethodColors(methodCodes)
    % Keep method colors stable across ablation and scaling figures.
    methodCodes = cellstr(string(methodCodes));
    colors = zeros(numel(methodCodes), 3);
    for methodIndex = 1:numel(methodCodes)
        switch upper(methodCodes{methodIndex})
            case 'M1'
                colors(methodIndex, :) = [0.0000, 0.4470, 0.7410];
            case 'M2'
                colors(methodIndex, :) = [0.9290, 0.6940, 0.1250];
            case 'M3'
                colors(methodIndex, :) = [0.4940, 0.1840, 0.5560];
            case 'M4'
                colors(methodIndex, :) = [0.4660, 0.6740, 0.1880];
            case 'M5'
                colors(methodIndex, :) = [0.3010, 0.7450, 0.9330];
            case 'M6'
                colors(methodIndex, :) = [0.8500, 0.3250, 0.0980];
            otherwise
                fallback = lines(numel(methodCodes));
                colors(methodIndex, :) = fallback(methodIndex, :);
        end
    end
end

function q = rowPercentileIgnoringNaN(values, pct)
    q = nan(size(values, 1), 1);
    for rowId = 1:size(values, 1)
        q(rowId) = TpIMFunctions.vectorPercentileIgnoringNaN(values(rowId, :), pct);
    end
end

function q = vectorPercentileIgnoringNaN(values, pct)
    values = values(:);
    values = sort(values(isfinite(values)));
    if isempty(values)
        q = NaN;
        return;
    end
    if isscalar(values)
        q = values(1);
        return;
    end
    pct = min(max(pct, 0), 100);
    pos = 1 + (numel(values) - 1) * pct / 100;
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        q = values(lo);
    else
        q = values(lo) + (pos - lo) * (values(hi) - values(lo));
    end
end

function targetIdx = getPrimaryTargetIndex(alphaTargets, primaryTarget)
    alphaTargets = alphaTargets(:)';
    [targetGap, targetIdx] = min(abs(alphaTargets - primaryTarget));
    if isempty(targetIdx) || ~isfinite(targetGap)
        error('TpIM:NoAlphaTargets', 'efficiencyAlphaTargets must contain at least one finite alpha target.');
    end
    if targetGap > 1e-9
        warning('TpIM:PrimaryTargetAdjusted', ...
            'primaryAlphaTarget %.3f is not in efficiencyAlphaTargets; using target %d (alpha %.3f).', ...
            primaryTarget, targetIdx, alphaTargets(targetIdx));
    end
end

function targetEval = getScalingTargetEvaluation(scalingTable, rowIdx, targetIdx)
    varNames = string(scalingTable.Properties.VariableNames);
    sequentialCol = sprintf('EvalToTarget%d', targetIdx);
    if any(varNames == sequentialCol)
        targetEval = scalingTable.(sequentialCol)(rowIdx);
        return;
    end

    error('TpIMFunctions:MissingTargetColumn', ...
        'Missing target-evaluation column "%s". Available columns: %s', ...
        sequentialCol, strjoin(cellstr(varNames), ', '));
end

function windowPoints = getConvergenceSmoothingWindow(nGridPoints, cfg)
    windowPoints = 1;
    if ~isfield(cfg, 'convergencePlot') || ~isfield(cfg.convergencePlot, 'smoothWindowFraction')
        return;
    end
    smoothFraction = cfg.convergencePlot.smoothWindowFraction;
    if isempty(smoothFraction) || ~isfinite(smoothFraction) || smoothFraction <= 0
        return;
    end
    windowPoints = max(1, round(nGridPoints * smoothFraction));
    if mod(windowPoints, 2) == 0
        windowPoints = windowPoints + 1;
    end
    windowPoints = min(windowPoints, nGridPoints);
end

function ySmooth = smoothConvergenceCurve(y, windowPoints)
    ySmooth = y(:);
    if windowPoints <= 1 || numel(ySmooth) <= 2
        return;
    end
    valid = isfinite(ySmooth);
    if ~any(valid)
        return;
    end
    x = (1:numel(ySmooth))';
    if ~all(valid)
        ySmooth(~valid) = interp1(x(valid), ySmooth(valid), x(~valid), 'linear', 'extrap');
    end
    yMin = min(ySmooth);
    yMax = max(ySmooth);
    ySmooth = movmean(ySmooth, windowPoints, 'Endpoints', 'shrink');
    ySmooth = cummax(ySmooth);
    ySmooth = min(max(ySmooth, yMin), yMax);
end

function switchEval = getStageSwitchEvaluations(stageLog)
    switchEval = [];
    if isempty(stageLog)
        return;
    end
    if iscell(stageLog)
        if isempty(stageLog) || numel(stageLog) < 2
            return;
        end
        stageItems = [stageLog{:}];
    else
        stageItems = stageLog;
    end
    if numel(stageItems) < 2 || ~isfield(stageItems, 'evalOffsetAfter')
        return;
    end
    switchEval = [stageItems(1:end-1).evalOffsetAfter]';
    switchEval = unique(switchEval(isfinite(switchEval) & switchEval > 0), 'stable');
end

function gridValues = resampleBestTrace(traceEval, traceValue, evalGrid)
    traceEval = traceEval(:);
    traceValue = traceValue(:);
    [traceEval, uniqueIdx] = unique(traceEval, 'stable');
    traceValue = traceValue(uniqueIdx);
    gridValues = nan(size(evalGrid));
    for i = 1:numel(evalGrid)
        idx = find(traceEval <= evalGrid(i), 1, 'last');
        if isempty(idx)
            gridValues(i) = traceValue(1);
        else
            gridValues(i) = traceValue(idx);
        end
    end
end

function [V, C, cellStatus] = buildBoundedVoronoi(Seed, Vcub, tol)
%%
% Returns the Voronoi vertices V and the Voronoi cells C of the Voronoi diagram 
% for the 3D points in the matrix seeds contained inside the rectangular cuboid 
% defined by the 8 corners positions 'Vcub'.
%
% tol input is the number of decimal of the new vertices generate on the domain edges (round(M,tol)) to avoid
% round-off errors (equal 12 by default)
%
% cellStatus is 0 outside the box, 1 for an uncut interior cell, and 2 for
% a cell clipped by the cuboid boundary.

if nargin < 3
    tol=12;
end
eps=10^(-tol);

% ========================================================
%Verify if 'Vcub' forms a rectangular cuboid
CUB0=min(Vcub); CUB1=max(Vcub);
for x=[CUB0(1) CUB1(1)];for y=[CUB0(2) CUB1(2)];for z=[CUB0(3) CUB1(3)] %#ok<ALIGN>
if any([x y z]~=0)
    idx=find(Vcub(:,1)==x & Vcub(:,2)==y & Vcub(:,3)==z, 1);
    if isempty(idx)
        error('#1 your input "Vcub" does not form a rectangular cuboid')
    end
end
end;           end;           end

   
% ========================================================
% Add seeds to avoid infinite cells inside the borders during the voronoi diagrams
% Cuboid boundary sampling points.

dim=max([Seed; Vcub])-min([Seed; Vcub]);
mid=(max([Seed; Vcub])+min([Seed; Vcub]))/2; 
LIMITER=zeros(3*9-1,3); n=0;
for i=[-1 0 1];for j=[-1 0 1];for k=[-1 0 1] %#ok<ALIGN>
if any([i j k]~=0)
    n=n+1;
    LIMITER(n,:)=3*(dim.*[i j k])/2+mid;
end
end;           end;           end
% Optional diagnostic: display boundary points used by the clipping step.
%

% ========================================================
% Classical voronoi diagram without limits
[V0,c]=voronoin([Seed; LIMITER]); % Compute vertices and cell connectivity.
C=c(1:size(Seed,1));
% V0=round(V0,tol);


% ========================================================
% Find the polyhedron that contains the corners and add it to their vertices
for j=1:size(Vcub,1)
    idx=1;
    MIN=norm(Vcub(j,:)-Seed(1,:));
    for i=2:size(Seed,1)
        tamp=norm(Vcub(j,:)-Seed(i,:));
        if tamp<MIN
            MIN=tamp;
            idx=i;
        elseif tamp==MIN
            idx=[idx i];
        end
    end
    M=Vcub(j,:);
    idxV=find(V0(:,1)==M(1) & V0(:,2)==M(2) & V0(:,3)==M(3));
    if isempty(idxV)
        V0=[V0; M]; idxV=size(V0,1);
    elseif length(idxV)>1
        error('#2 There are duplicates in V')
    end
    for k=idx
        C{k}=[C{k} idxV];
    end
end


% ========================================================
% Find all the points outside the box (==1 for inside)
vTst=zeros(size(V0,1),1);
for pt=1:size(V0,1)
    if all(V0(pt,:)>=CUB0) && all(V0(pt,:)<=CUB1)
        vTst(pt)=1;
    end
end


% ========================================================
% ========================================================
% Cut all the polyhedron on the edges of the box
M=[0 0 0];
% List of box faces
      %i0 i1 i2 M(i0)
listF=[1 2 3 CUB0(1); 2 1 3 CUB0(2); 3 2 1 CUB0(3);
       1 2 3 CUB1(1); 2 1 3 CUB1(2); 3 2 1 CUB1(3);];
% List of box lines
      %i0 i0dim i1 i1dim i2
listL=[1 CUB0(1) 2 CUB0(2) 3; 1 CUB0(1) 3 CUB0(3) 2; 3 CUB0(3) 2 CUB0(2) 1;
       1 CUB1(1) 2 CUB0(2) 3; 1 CUB1(1) 3 CUB0(3) 2; 3 CUB1(3) 2 CUB0(2) 1;
       1 CUB0(1) 2 CUB1(2) 3; 1 CUB0(1) 3 CUB1(3) 2; 3 CUB0(3) 2 CUB1(2) 1; 
       1 CUB1(1) 2 CUB1(2) 3; 1 CUB1(1) 3 CUB1(3) 2; 3 CUB1(3) 2 CUB1(2) 1];
cellStatus=zeros(length(C),1);

for k=1:length(C)
    Ck=C{k};

    if all(Ck~=1) && all(vTst(Ck)==1) % All vertices are already inside the box
        cellStatus(k)=1; %disp([1 k])

    elseif all(Ck~=1) && ... % The polyhedron is finite but has parts outside the box
            ( (any(vTst(Ck)==1) && any(vTst(Ck)==0)) || all(vTst(Ck)==0) )

        cellStatus(k)=2; %disp([2 k])
        Vk = V0(Ck,:);
        C{k}=Ck(vTst(Ck)==1);
        Fk=convhull(Vk);

        % ========================================================
        % for each face of the polyhedron look for :
        for f=1:size(Fk,1)

            % Intersection between cube's edges and polyhedron's face f
            Ap=Vk(Fk(f,1),:); Bp=Vk(Fk(f,2),:); Cp=Vk(Fk(f,3),:);
            for l=1:12
                i1=listL(l,1); i2=listL(l,3); i3=listL(l,5);
                Al=zeros(1,3); Al(i1)=listL(l,2); Al(i2)=listL(l,4); Al(i3)=CUB0(i3);
                Bl=zeros(1,3); Bl(i1)=listL(l,2); Bl(i2)=listL(l,4); Bl(i3)=CUB1(i3);
                [M,TST]=TpIMFunctions.intersectSegmentWithTriangle(Ap,Bp,Cp,Al,Bl,eps);
                if TST
                    M=round(M,tol);
                    idx=find(V0(:,1)==M(1) & V0(:,2)==M(2) & V0(:,3)==M(3));
                    if isempty(idx)
                        V0=[V0; M]; idx=size(V0,1);
                    elseif length(idx)>1
                        error('#3 There are duplicates in V')
                    end
                    C{k}=[C{k} idx];
                end
            end

            % Intersection between cube's face p and poly's edge v
            for v=1:3
                Ai=Fk(f,v); Bi=Fk(f,mod(v+1-1,3)+1);
                A=Vk(Ai,:); B=Vk(Bi,:);
                for p=1:6
                    i0=listF(p,1); i1=listF(p,2); i2=listF(p,3);
                    if (B(i0)-A(i0))~=0
                        M(i0)=listF(p,4); t=(M(i0)-A(i0))/(B(i0)-A(i0));
                        M(i1)=A(i1)+t*(B(i1)-A(i1));
                        M(i2)=A(i2)+t*(B(i2)-A(i2));
                        if (all(M>=(min([A; B])-eps)) && all(M<=(max([A; B])+eps)))
                            if (all(M>=(CUB0-eps)) && all(M<=(CUB1+eps)))
                                M=round(M,tol);
                                idx=find(V0(:,1)==M(1) & V0(:,2)==M(2) & V0(:,3)==M(3));
                                if isempty(idx)
                                    V0=[V0; M]; idx=size(V0,1);
                                elseif length(idx)>1
                                    error('#4 There are duplicates in V')
                                end
                                C{k}=[C{k} idx];
                            end
                        end
                    end
                end
            end

        end

        %Suppress all unecessary multiple points placed on a single same edge
        C{k}=unique(C{k});
        Vk=V0(C{k},:);
        LIST=[];
        for i0=1:size(Vk,1)
            l=[1:(i0-1) (i0+1):size(Vk,1)];
            x=(Vk(l,:)-Vk(i0,:))./vecnorm(Vk(l,:)-Vk(i0,:),2,2);
            t=acosd(round(x*x',tol));
            if all(t<180)
                LIST=[LIST i0];
            end
        end
        %     disp(C{k})
        %     disp(Vk)
        %     disp(LIST)
        C{k}=C{k}(LIST);

    else % All the remaining infinite cells
        C{k}=[]; cellStatus(k)=0;
    end
end

% % Round the vertices
% Vrnd=round(V0,tol);
Vrnd=V0;

% Eleminate vertices duplicates
iV=sort(unique([C{:}]));
V=sortrows(unique(Vrnd(iV,:),'rows'),[3 2 1]);

cellStatus = zeros(length(C),1);
% Supress all the unused vertices and rearrange their numbers
for k=1:length(C)    
if ~isempty(C{k}) && size(C{k},2)>3
    C{k}=sort(unique(C{k}));
    Vk = Vrnd(C{k},:);
    % If all the vertices are outside, or all on the edges the box, the cell 
    % is none existent.
    if any(all(Vk<=CUB0)) || any(all(Vk>=CUB1))
        C{k}=[];
        cellStatus(k)=0;
    else
        Fk = convhull(Vk); 

        C{k}=zeros(1,size(Vk,1));
        for i=1:size(Vk,1)
            idx=find(V(:,1)==Vk(i,1) & V(:,2)==Vk(i,2) & V(:,3)==Vk(i,3));
            if length(idx)>1
                error('#5 There is an issue (it shouldn''t be empty)')
            end
            C{k}(i)=idx(1);
        end
    end

else
    cellStatus(k)=1;
end
end
C(cellStatus==1,:)=[];
end

function [M,TST]=intersectSegmentWithTriangle(Ap,Bp,Cp,Al,Bl,eps)
    % Ap,Bp,Cp : three points defining the plan
    % Al,Bl : two points defining the line
    % eps : round-off error tolerance
    % M : intersection point
    % TST: Test variable: true if M is inside (edges included), false if not.
    
    abc = cross(Bp-Ap, Cp-Ap);
        N=abc/norm(abc); % Normal to the plan
    d = -sum(Ap.*N); 
    
    U=Bl-Al; U=U/norm(U); % Line vector
    
    div=sum(N.*U);
    % Colinearity test
    if abs(div)<10^(-8) % To avoid decimal error that could happen
    	M=[]; TST=false;
    else
        % Plan equation: sum(N.*M)+d=0
        % Line equation: M-Al=t*U
        t=-(sum(N.*Al)+d)/div;
        M=t*U+Al;
        
        % Test if M is inside the triangle
        TriTst=TpIMFunctions.isPointInTriangle(Ap,Bp,Cp,M,eps);
        % Test if M is between Al and Bl
        LinTst=(all(M >=(min([Al;Bl])-eps)) && all(M <=(max([Al;Bl])+eps)));
        
        TST = TriTst && LinTst;
    end
end

function TST=isPointInTriangle(A,B,C,M,eps)
% Test if M is inside (edges included) the triangle formed by the three
% points A, B, C.
% M must already be in the plan ABC or it doesn't work
% eps : round-off error tolerance

% In 2D, it is easy to check this: the cross product AMB, BMC, CMA must be
% of identical signs.
% To extend this idea in 3D, I project the triangle on the three plans 
% XY, YZ and ZX and does the 2D test (I migh be wrong but it seems to work
% really well).

iL=[1 2; 2 3; 3 1]; 
TST= true;
for l=1:3    
    p=iL(l,:); a=A(p); b=B(p); c=C(p); m=M(p);
    ma=(a-m)/vecnorm(a-m); mb=(b-m)/vecnorm(b-m); mc=(c-m)/vecnorm(c-m); 
    ca=(a-c)/vecnorm(a-c); cb=(b-c)/vecnorm(b-c);

    % Test if a,b,c are not colinear
    vTst=ca(1)*cb(2)-ca(2)*cb(1);
    if abs(vTst) > eps % To avoid decimal error that could happen 
        v=[ ma(1)*mb(2)-ma(2)*mb(1)
            mb(1)*mc(2)-mb(2)*mc(1)
            mc(1)*ma(2)-mc(2)*ma(1) ];
        v(abs(v)<eps)=0;
        TST = TST && ( ...
                    (sign(v(1)*v(2))==1 || sign(v(1)*v(2))==0) ...
                    && ...
                    (sign(v(1)*v(3))==1 || sign(v(1)*v(3))==0) ...
                    && ...
                    (sign(v(2)*v(3))==1 || sign(v(2)*v(3))==0) ...
                    );
    else % If the projection forms a line
        TST = TST && ( ...    
              (all(m >=(min([a;b;c])-eps)) && all(m <=(max([a;b;c])+eps))) ...
                    );
    end
end

end

end
end
