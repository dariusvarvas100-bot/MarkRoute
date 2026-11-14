/* Run by test_markroute.py with the released engine and Python replay values. */
(() => {
  const fields = ['modelFt', 'baselineFt', 'startingFt', 'correctionFt'];
  const tolerance = 1e-7;
  let initial = 0;
  let afterReveal = 0;
  let blinded = 0;
  let maximumDifference = 0;
  let updatedScenarios = 0;
  let attributions = 0;
  let changedAttributions = 0;
  let counterfactuals = 0;
  let changedCounterfactuals = 0;
  let maximumRemovalDifference = 0;

  function assertEstimates(estimate, expected, context) {
    for (const [key, field] of fields.entries()) {
      const actual = estimate[field];
      const reference = expected[key];
      if (reference === null) {
        if (actual !== null) throw new Error(`${context} surveyed point has an estimate`);
        continue;
      }
      const delta = Math.abs(actual - reference);
      if (!Number.isFinite(actual) || !Number.isFinite(reference) || delta > tolerance) {
        throw new Error(`${context} ${field} differs by ${delta}`);
      }
      maximumDifference = Math.max(maximumDifference, delta);
    }
  }

  function close(actual, expected, context) {
    if (!Number.isFinite(actual) || !Number.isFinite(expected) || Math.abs(actual - expected) > tolerance) {
      throw new Error(`${context}: ${actual} does not match ${expected}`);
    }
  }

  function checkReadingCounterfactual(sites, result, details, context) {
    const {target, removed, fullFt, withoutFt} = details;
    const model = MARKROUTE_DATA.estimator;
    const frozen = Object.freeze(sites.map(site => Object.freeze({...site})));
    const before = JSON.stringify(frozen);
    if (frozen[target].surveyed || !frozen[removed].surveyed) throw new Error(`${context}: infeasible removal`);
    const explanation = MarkRouteEngine.explainSurveyInfluence(frozen, model, target);
    if (!explanation.available || !explanation.contributions.some(entry => entry.index === removed)) {
      throw new Error(`${context}: remove only a listed survey reading`);
    }
    close(explanation.modelFt, fullFt, `${context} full estimate against Python`);
    close(result[target].modelFt, fullFt, `${context} replay estimate against Python`);

    const omitted = Object.freeze(frozen.map((site, i) => Object.freeze({...site, surveyed: site.surveyed && i !== removed})));
    const blind = Object.freeze(omitted.map(site => site.surveyed ? site : Object.freeze({...site, actualFt: null})));
    const blindedBefore = JSON.stringify(blind);
    const preview = MarkRouteEngine.recompute(blind, model);
    const withArchiveValues = MarkRouteEngine.recompute(omitted, model);
    close(preview[target].modelFt, withoutFt, `${context} without reading against Python`);
    close(preview[target].modelFt - result[target].modelFt, withoutFt - fullFt, `${context} displayed signed change`);
    if (preview[target].modelFt !== withArchiveValues[target].modelFt) {
      throw new Error(`${context}: hidden elevations, including the simulated removal, changed the estimate`);
    }
    if (JSON.stringify(frozen) !== before || JSON.stringify(blind) !== blindedBefore) {
      throw new Error(`${context}: preview mutated the replay, or its masked copy`);
    }
    close(MarkRouteEngine.recompute(frozen, model)[target].modelFt, fullFt, `${context} restore full estimate`);
    maximumRemovalDifference = Math.max(maximumRemovalDifference, Math.abs(preview[target].modelFt - withoutFt));
    if (Math.abs(withoutFt - fullFt) > tolerance) changedCounterfactuals++;
    counterfactuals++;
  }

  function checkSyntheticAttribution() {
    const makeSite = (id, offset, height, surveyed, environment = 'Riverine') => ({
      id, lat: 42 + offset * .009, lon: -78 + (offset % 2) * .003, environment,
      surveyed, actualFt: height, groundFt: 10 + offset, pointFt: 11 + offset,
      reliefFt: 2 + offset, waterbody: 'test river'
    });
    const sites = [makeSite('query', 0, 95, false), makeSite('a', 1, 31, true),
      makeSite('b', 2, 22, true), makeSite('c', 3, 11, true), makeSite('d', 4, 47, true),
      makeSite('fixed', 5, 15, true), makeSite('other', 6, 65, true, 'Coastal')];
    const model = MARKROUTE_DATA.estimator;
    const explanation = MarkRouteEngine.explainSurveyInfluence(sites, model, 0);
    if (!explanation.available || explanation.fixedCount !== 2) throw new Error('Four-player fixture unavailable');
    const expectedIndices = [1, 2, 3, 4];
    if (String(explanation.contributions.map(item => item.index)) !== String(expectedIndices)) {
      throw new Error('Nearest four local surveys were not selected');
    }
    const fixed = sites.map((site, i) => expectedIndices.includes(i) ? {...site, surveyed: false} : site);
    close(explanation.startingFt, MarkRouteEngine.recompute(fixed, model)[0].modelFt, 'Synthetic fixed surveys');

    // Independently average the marginal effect over all survey reveal orders.
    const sums = new Map(expectedIndices.map(index => [index, 0]));
    let permutations = 0;
    function visit(remaining, included, previous) {
      if (!remaining.length) { permutations++; return; }
      for (const index of remaining) {
        const updated = new Set([...included, index]);
        const view = fixed.map((site, i) => updated.has(i) ? {...site, surveyed: true} : site);
        const estimate = MarkRouteEngine.recompute(view, model)[0].modelFt;
        let paths = 1;
        for (let k = 2; k < remaining.length; k++) paths *= k;
        sums.set(index, sums.get(index) + paths * (estimate - previous));
        visit(remaining.filter(other => other !== index), updated, estimate);
      }
    }
    visit(expectedIndices, new Set(), explanation.startingFt);
    for (const contribution of explanation.contributions) {
      close(contribution.contributionFt, sums.get(contribution.index) / permutations, 'Permutation enumeration');
    }
    for (const action of [
      MarkRouteEngine.explainSurveyInfluence(sites, model, 2),
      MarkRouteEngine.explainSurveyInfluence(sites, model, 100),
      MarkRouteEngine.explainSurveyInfluence(sites.map((site, i) => i < 2 || i > 5 ? site : {...site, surveyed: false}), model, 0)
    ]) {
      if (action.available || !action.reason) throw new Error('Expected a plain unavailable reason');
    }
  }

  checkSyntheticAttribution();

  for (const event of MARKROUTE_DATA.events) {
    for (const replay of event.scenarios) {
      const context = `${event.name}, seed ${replay.seed}`;
      const baseline = MarkRouteEngine.recompute(replay.sites);
      // The fixture contains archive truth for the demo Reveal button. Forecast code
      // may use it only after the site is marked surveyed.
      const masked = replay.sites.map(site => site.surveyed ? site : {...site, actualFt: null});
      const retest = MarkRouteEngine.recompute(masked);
      const reveal = replay.sites.map((site, i) => i === replay.addedSurvey ? {...site, surveyed: true} : site);
      const next = MarkRouteEngine.recompute(reveal);
      if (!next[replay.addedSurvey].surveyed) throw new Error(`${context} failed to add the survey`);
      if (next.some((site, i) => !reveal[i].surveyed && Math.abs(site.modelFt - baseline[i].modelFt) > tolerance)) {
        updatedScenarios++;
      }
      for (const [index, site] of replay.sites.entries()) {
        assertEstimates(baseline[index], fields.map(field => site[field]), context);
        assertEstimates(retest[index], fields.map(field => baseline[index][field]), `${context} blinded`);
        assertEstimates(next[index], replay.afterSurvey[index], `${context} after reveal`);
        if (!site.surveyed) { initial++; blinded++; }
        if (!reveal[index].surveyed) afterReveal++;
      }

      const model = MARKROUTE_DATA.estimator;
      const target = replay.sites.findIndex((site, i) => !site.surveyed && i !== replay.addedSurvey &&
        replay.sites.filter(point => point.surveyed && point.environment === site.environment).length >= 2);
      if (target < 0) throw new Error(`${context} has no testable attribution`);
      const influence = MarkRouteEngine.explainSurveyInfluence(replay.sites, model, target);
      const maskedInfluence = MarkRouteEngine.explainSurveyInfluence(masked, model, target);
      if (!influence.available || JSON.stringify(influence) !== JSON.stringify(maskedInfluence)) {
        throw new Error(`${context} attribution changed with hidden labels`);
      }
      close(influence.modelFt, baseline[target].modelFt, `${context} deployed estimate`);
      const contributionSum = influence.contributions.reduce((sum, item) => sum + item.contributionFt, 0);
      close(influence.startingFt + contributionSum, influence.modelFt, `${context} conservation`);
      const afterInfluence = MarkRouteEngine.explainSurveyInfluence(reveal, model, target);
      close(afterInfluence.modelFt, next[target].modelFt, `${context} deployed estimate after reveal`);
      const oldContributions = influence.contributions.map(item => [item.id, item.contributionFt]);
      const newContributions = afterInfluence.contributions.map(item => [item.id, item.contributionFt]);
      if (JSON.stringify(oldContributions) !== JSON.stringify(newContributions)) changedAttributions++;
      attributions++;
      for (const details of replay.readingCounterfactuals) {
        const added = details.step === 'afterAdd';
        checkReadingCounterfactual(added ? reveal : replay.sites, added ? next : baseline,
          details, `${context} ${details.step} removing ${details.removed}`);
      }
    }
  }
  if (!updatedScenarios) throw new Error('The new measurements never changed an estimate');
  if (!changedAttributions) throw new Error('The new measurements never changed an attribution');
  if (!changedCounterfactuals) throw new Error('Withholding a reading never changed the estimate');
  const summary = JSON.stringify({initial, afterReveal, blinded, updatedScenarios, attributions, changedAttributions,
    counterfactuals, changedCounterfactuals, maximumRemovalDifferenceFt: maximumRemovalDifference,
    maximumDifferenceFt: maximumDifference});
  if (typeof print === 'function') print(summary);
  else console.log(summary);
})();
