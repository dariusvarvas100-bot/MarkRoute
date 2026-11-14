/* The browser uses the same features and numeric tree splits as the Python replay. */
const MarkRouteEngine = (() => {
  function sum(values) {
    return values.reduce((total, value) => total + value, 0);
  }

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function treeValue(tree, values) {
    let node = tree[0];
    while (node.length !== 1) {
      const [feature, threshold, left, right] = node;
      node = tree[values[feature] <= threshold ? left : right];
    }
    return node[0];
  }

  function prediction(site, observed, model) {
    const local = observed.filter(point => point.environment === site.environment);
    const eligible = local.length ? local : observed;
    const peers = eligible.map((point, index) => ({
      point, index, distance: Math.hypot(site.x - point.x, site.y - point.y)
    })).sort((a, b) => a.distance - b.distance || a.index - b.index).slice(0, 12);

    const coast = site.environment === 'Coastal';
    const weights = peers.map(peer => 1 / (0.25 + peer.distance) ** 1.5);
    const scale = sum(weights);
    const w = weights.map(value => value / scale);
    const distances = peers.map(peer => site.groundFt - peer.point.groundFt);
    const elevations = peers.map(peer => peer.point.actualFt);
    const avg = sum(elevations.map((value, i) => value * w[i]));
    const dz = sum(distances.map((value, i) => value * w[i]));
    const depths = peers.map(({point}) => point.actualFt - point.groundFt);
    const terrain = site.groundFt + median(depths);
    const slope = sum(elevations.map((y, i) => w[i] * (y - avg) * (dz - distances[i]))) /
      (sum(distances.map((d, i) => w[i] * (d - dz) ** 2)) + 4);
    const baseline = avg + Math.max(0, Math.min(1.2, slope)) * dz;
    const base = coast ? avg : terrain;
    const pair = peers.map(({point, distance}, i) => [
      Math.log1p(distance), Math.asinh(distances[i] / 15), Math.asinh(depths[i] / 10),
      Number(site.waterbody !== '' && site.waterbody === point.waterbody), Number(coast),
      Math.log1p(site.reliefFt), Math.log1p(point.reliefFt),
      Math.asinh((site.pointFt - site.groundFt) / 5), Math.asinh((point.pointFt - point.groundFt) / 5),
      Math.asinh((site.x - point.x) / 5), Math.asinh((site.y - point.y) / 5)
    ]);
    const pooled = pair[0].map((_, column) => sum(pair.map((row, i) => w[i] * row[column])));
    const deviations = [elevations[0], avg, terrain, baseline].map(value => Math.asinh((value - base) / 8));
    const features = [...pair[0], ...(pair[1] ?? pair[0]), ...pooled, ...deviations,
      Math.log1p(peers.length), Math.asinh(site.groundFt / 50)];
    const quantized = features.map(value => Math.round(value * 1e6));
    const residual = model.intercept + sum(model.trees.map(tree => treeValue(tree, quantized)));
    return {modelFt: base + residual, baselineFt: baseline, startingFt: base, correctionFt: residual};
  }

  function projectSites(sites) {
    const midLat = sum(sites.map(site => site.lat)) / sites.length;
    const midLon = sum(sites.map(site => site.lon)) / sites.length;
    const horizontal = 111.2 * Math.cos(midLat * Math.PI / 180);
    return sites.map(site => ({
      ...site, x: horizontal * (site.lon - midLon), y: 111.2 * (site.lat - midLat)
    }));
  }

  function recompute(sites, model = MARKROUTE_DATA.estimator) {
    const points = projectSites(sites);
    const known = points.filter(point => point.surveyed);
    if (known.length === 0) return sites;
    return sites.map((site, i) => site.surveyed ?
      {...site, modelFt: null, baselineFt: null, startingFt: null, correctionFt: null} : {...site, ...prediction(points[i], known, model)});
  }

  function explainSurveyInfluence(sites, model, targetIndex) {
    const target = sites[targetIndex];
    if (!target) return {available: false, reason: 'Select an unmeasured mark to explain its estimate.'};
    if (target.surveyed) return {available: false, reason: 'This mark already has a survey reading.'};

    const points = projectSites(sites);
    const query = points[targetIndex];
    const surveyed = points.map((point, index) => ({point, index})).filter(({point}) => point.surveyed);
    const local = surveyed.filter(({point}) => point.environment === query.environment);
    if (local.length < 2) {
      return {available: false, reason: 'At least two surveyed marks of this environment are needed to compare their influence.'};
    }
    if (local.some(({point}) => !Number.isFinite(point.actualFt))) {
      return {available: false, reason: 'A survey reading is missing from this replay.'};
    }

    // Keeping one local reading fixed makes every coalition use the same
    // environment-specific estimator, including the empty coalition.
    const varied = local.map(({point, index}) => ({
      point, index, distanceKm: Math.hypot(query.x - point.x, query.y - point.y)
    })).sort((a, b) => a.distanceKm - b.distanceKm || a.index - b.index)
      .slice(0, Math.min(4, local.length - 1));
    const position = new Map(varied.map(({index}, bit) => [index, bit]));
    const count = varied.length;
    const full = (1 << count) - 1;
    const estimates = [];
    const coalitionSizes = [];
    for (let mask = 0; mask <= full; mask++) {
      let size = 0;
      const observed = surveyed.filter(({index}) => {
        const bit = position.get(index);
        if (bit === undefined) return true;
        if (!(mask & (1 << bit))) return false;
        size++;
        return true;
      }).map(({point}) => point);
      coalitionSizes.push(size);
      estimates.push(prediction(query, observed, model).modelFt);
    }

    const factorial = [1, 1, 2, 6, 24];
    const contributions = varied.map(({point, index, distanceKm}, bit) => {
      const add = 1 << bit;
      let contributionFt = 0;
      for (let mask = 0; mask <= full; mask++) {
        if (mask & add) continue;
        const size = coalitionSizes[mask];
        const weight = factorial[size] * factorial[count - size - 1] / factorial[count];
        contributionFt += weight * (estimates[mask | add] - estimates[mask]);
      }
      return {id: point.id, index, observedFt: point.actualFt, distanceKm, contributionFt};
    });
    return {available: true, startingFt: estimates[0], modelFt: estimates[full], contributions,
      fixedCount: surveyed.length - count};
  }

  return {recompute, explainSurveyInfluence};
})();
